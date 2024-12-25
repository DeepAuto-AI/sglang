from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable

import torch
import tqdm
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import graph_capture

from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner, patch_model, clamp_position

if TYPE_CHECKING:
    from sglang.srt.model_executor.hip_model_runner import HiPModelRunner


class HiPCudaGraphRunner(CudaGraphRunner):

    def __init__(self, model_runner: "HiPModelRunner"):
        super().__init__(model_runner)

    def can_run(self, forward_batch: ForwardBatch):
        use_cached_mask = forward_batch.hip_use_cached_mask

        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = min(forward_batch.global_num_tokens), max(
                forward_batch.global_num_tokens
            )
            is_bs_supported = forward_batch.can_run_dp_cuda_graph and (
                (min_num_tokens == max_num_tokens and (max_num_tokens, use_cached_mask) in self.graphs)
                if self.disable_padding
                else max_num_tokens <= self.max_bs
            )
        else:
            is_bs_supported = (
                (forward_batch.batch_size, use_cached_mask) in self.graphs
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )
        return is_bs_supported and is_encoder_lens_supported

    def capture(self):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_bs = (
                tqdm.tqdm(self.capture_bs)
                if get_tensor_model_parallel_rank() == 0
                else self.capture_bs
            )
            for bs in capture_bs:
                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    bs,
                    self.model_runner.tp_group,
                ) as forward:
                    for use_cached_mask in [False, True]:
                        (
                            graph,
                            output_buffers,
                        ) = self.capture_one_batch_size(bs, forward, use_cached_mask)
                        self.graphs[(bs, use_cached_mask)] = graph
                        self.output_buffers[(bs, use_cached_mask)] = output_buffers

    def capture_one_batch_size(self, bs: int, forward: Callable, hip_use_cached_mask: bool = False):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        # Common inputs
        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:bs]
        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None

        seq_lens_sum = seq_lens.sum().item()
        mrope_positions = self.mrope_positions[:, :bs]

        if self.enable_dp_attention:
            global_num_tokens = [bs] * self.tp_size
            gathered_buffer = self.gathered_buffer[: bs * self.tp_size]
        else:
            global_num_tokens = None
            gathered_buffer = None

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            encoder_lens,
        )

        # Run and capture
        def run_once():
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=bs,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                hip_metadata_cache_pool=self.model_runner.hip_metadata_cache_pool,
                hip_use_cached_mask=hip_use_cached_mask,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=seq_lens_sum,
                encoder_lens=encoder_lens,
                return_logprob=False,
                top_logprobs_nums=[0] * bs,
                positions=clamp_position(seq_lens),
                mrope_positions=mrope_positions,
                global_num_tokens=global_num_tokens,
                gathered_buffer=gathered_buffer,
            )
            logits_output = forward(input_ids, forward_batch.positions, forward_batch)
            return logits_output.next_token_logits

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        self.graph_memory_pool = graph.pool()
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        raw_bs = forward_batch.batch_size

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs, max(forward_batch.global_num_tokens)
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_bs].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_bs].copy_(forward_batch.out_cache_loc)
        if self.is_encoder_decoder:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)
        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            self.encoder_lens,
        )

        # Replay
        key = (bs, forward_batch.hip_use_cached_mask)
        self.graphs[key].replay()
        next_token_logits = self.output_buffers[key][:raw_bs]

        # Extract logprobs
        if forward_batch.return_logprob:
            logits_metadata = LogitsMetadata(
                forward_mode=ForwardMode.DECODE,
                top_logprobs_nums=forward_batch.top_logprobs_nums,
            )
            next_token_logprobs = (
                LogitsProcessor.compute_temp_top_p_normalized_logprobs(
                    next_token_logits, logits_metadata
                )
            )
            logits_output = LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                next_token_logprobs=next_token_logprobs,
            )
            return_top_logprob = any(x > 0 for x in forward_batch.top_logprobs_nums)
            if return_top_logprob:
                (
                    logits_output.output_top_logprobs_val,
                    logits_output.output_top_logprobs_idx,
                ) = LogitsProcessor.get_top_logprobs(
                    next_token_logprobs, logits_metadata
                )[
                    2:4
                ]
        else:
            logits_output = LogitsProcessorOutput(
                next_token_logits=next_token_logits,
            )

        return logits_output
