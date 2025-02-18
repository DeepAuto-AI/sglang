from __future__ import annotations

import os

"""
HiP Attention Backend for SGLang
https://arxiv.org/pdf/2406.09827
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool

if TYPE_CHECKING:
    from hip.models.hip_attention.gen3 import HiPAttentionConfig

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

logger = logging.getLogger(__name__)


class HiPRadixAttentionBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        from hip.models.hip_attention.gen3 import forward_paged_hip

        self.forward_paged_hip = forward_paged_hip

        self.hip_config: HiPAttentionConfig = (
            model_runner.server_args.hip_attention_config
        )
        self.is_offload_enabled = model_runner.server_args.enable_hip_offload

        self.max_context_len = model_runner.model_config.context_len

        self.tp_rank = model_runner.tp_rank

        # NOTE: query caching: this is quite temporary one.
        self.q_buffers = [
            torch.zeros(
                (
                    1,
                    self.hip_config.block_sparse_block_size_q,
                    model_runner.model_config.num_attention_heads
                    // model_runner.tp_size,
                    model_runner.model_config.head_dim,
                ),
                device=torch.device(model_runner.device),
                dtype=model_runner.dtype,
            )
            for _ in range(model_runner.model_config.num_hidden_layers)
        ]
        # NOTE: disable q caching
        self.q_buffers = None

        # NOTE: sliding window indices: this is quite temporary one.
        diag_info_path = os.getenv("HIP_DIAG_INFO", None)
        if diag_info_path is not None:
            self.diag_sliding_window_indices = []
            for layer_id in range(model_runner.model_config.num_hidden_layers):
                require_dense = layer_id in self.hip_config.dense_layers
                if len(self.hip_config.layers) == 2:
                    layer_config = self.hip_config.layers[0 if require_dense else 1]
                else:
                    layer_config = self.hip_config.layers[layer_id]
                exclude_window_size = layer_config.sliding_window_size // 2
                diag_sliding_window_range = 131072
                diag_sliding_window_size = 8192 if require_dense else 4096

                chunk_size = layer_config.stages[-1].stage_chunk_size
                block_size_q = layer_config.stages[-1].stage_block_size_q

                diag_info = torch.load(
                    diag_info_path, map_location=model_runner.device
                )["prob_avg"][
                    layer_id
                ]  # type: torch.Tensor
                assert diag_info.ndim == 2

                diag_info = diag_info.sum(dim=0, keepdim=True).expand_as(diag_info)
                diag_info[
                    :, : max(0, diag_info.shape[-1] - diag_sliding_window_range)
                ].fill_(-32000.0)
                diag_info[:, -exclude_window_size:].fill_(-32000.0)
                diag_info = torch.nn.functional.max_pool1d(
                    diag_info.unsqueeze(1),
                    kernel_size=max(chunk_size, block_size_q) + 1,
                    stride=1,
                    padding=max(chunk_size, block_size_q) // 2,
                ).squeeze(1)
                diag_info = diag_info[:, ::chunk_size]
                _, indices = diag_info.topk(
                    k=diag_sliding_window_size // chunk_size, dim=-1
                )

                head_per_tp = (
                    model_runner.model_config.num_attention_heads
                    // model_runner.tp_size
                )
                idx_head_start = head_per_tp * model_runner.tp_rank
                idx_head_end = head_per_tp * (model_runner.tp_rank + 1)
                self.diag_sliding_window_indices.append(
                    (indices[idx_head_start:idx_head_end] - diag_info.shape[-1])
                    * chunk_size
                )
        else:
            self.diag_sliding_window_indices = None

    def push_q_buffer(self, q: torch.Tensor, layer_id: int, batch_size: int):
        if self.q_buffers is None:
            return
        assert batch_size == 1
        q = q.unsqueeze(0)
        layer_q_buffer = self.q_buffers[layer_id]
        q_buffer = torch.cat([layer_q_buffer, q[:, -layer_q_buffer.shape[1] :]], dim=1)
        layer_q_buffer.copy_(q_buffer[:, -layer_q_buffer.shape[1] :])

    def get_q_buffer(self, layer_id: int, batch_size: int) -> torch.Tensor:
        if self.q_buffers is not None:
            assert batch_size == 1
            return self.q_buffers[layer_id].flatten(0, 1)
        else:
            return None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int):
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        pass

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if not self.is_offload_enabled:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            offload_cache = None

        else:  # Offloading enabled
            assert isinstance(
                forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
            )
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, async_copy=True, push_to_gpu_cache=False
                    )
            k_cache = v_cache = None
            offload_cache = None

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        self.push_q_buffer(
            q_reshaped, layer_id=layer.layer_id, batch_size=forward_batch.batch_size
        )

        # Output tensor
        o = torch.empty_like(q_reshaped)

        start_len = 0
        decoding_reqs = []
        decoding_reqs_positions = []
        for idx_batch, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_positions.append(start_len)

            else:
                if not self.is_offload_enabled:
                    k_chunk = v_chunk = None
                    offloading_metadata = None

                else:  # Offloading enabled
                    k_chunk, v_chunk, offloading_metadata = (
                        forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                            layer_id=layer.layer_id,
                            batch_id=idx_batch,
                            cache_k=k[start_len : start_len + seq_len].unsqueeze(0),
                            cache_v=v[start_len : start_len + seq_len].unsqueeze(0),
                        )
                    )
                    offload_cache = k_cache = v_cache = None

                o_req, _ = self.forward_paged_hip(
                    query=q_reshaped[start_len : start_len + seq_len],
                    sm_scale=layer.scaling,
                    batch_size=1,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions[start_len : start_len + seq_len],
                    seq_lens=forward_batch.seq_lens[idx_batch : idx_batch + 1],
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices[
                        idx_batch : idx_batch + 1
                    ],
                    rope_cos=layer.rope_cos,
                    rope_sin=layer.rope_sin,
                    layer_id=layer.layer_id,
                    logit_cap=layer.logit_cap,
                    orig_context_len=layer.orig_context_len,
                    max_context_len=self.max_context_len,
                    is_prefill=True,
                    hip_config=self.hip_config,
                    k=k_chunk,
                    v=v_chunk,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                        if self.is_offload_enabled
                        else None
                    ),
                    offloading_metadata=offloading_metadata,
                    is_decode=False,
                    diag_sliding_window_indices=self.diag_sliding_window_indices,
                )

                o[start_len : start_len + seq_len] = o_req

            start_len += seq_len

        assert len(decoding_reqs) == 0

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        metadata = None
        if forward_batch.hip_metadata_cached_stages <= 0:
            metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                layer.layer_id,
                q.shape[0],
                forward_batch.batch_size,
                max(0, forward_batch.hip_metadata_cached_stages),
            )

        if not self.is_offload_enabled:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            offload_cache = None

        else:  # Offloading enabled
            assert isinstance(
                forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
            )
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, async_copy=False, push_to_gpu_cache=True
                    )

            k_cache = v_cache = None
            offload_cache, offloading_metadata = (
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            )

        self.push_q_buffer(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            layer_id=layer.layer_id,
            batch_size=forward_batch.batch_size,
        )
        q_for_masking = self.get_q_buffer(layer.layer_id, forward_batch.batch_size)

        o, metadata = self.forward_paged_hip(
            query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            sm_scale=layer.scaling,
            batch_size=forward_batch.batch_size,
            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,
            positions=forward_batch.positions,
            seq_lens=forward_batch.seq_lens,
            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            rope_cos=layer.rope_cos,
            rope_sin=layer.rope_sin,
            layer_id=layer.layer_id,
            logit_cap=layer.logit_cap,
            orig_context_len=layer.orig_context_len,
            max_context_len=self.max_context_len,
            is_prefill=False,
            hip_config=self.hip_config,
            cached_metadata=metadata,
            online_update_cache=(
                forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                if self.is_offload_enabled
                else None
            ),
            is_decode=True,
            query_for_mask=q_for_masking,
            diag_sliding_window_indices=self.diag_sliding_window_indices,
        )

        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
            layer_id=layer.layer_id,
            size=q.shape[0],
            batch_size=forward_batch.batch_size,
            metadata=metadata,
        )

        if self.is_offload_enabled:
            offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
