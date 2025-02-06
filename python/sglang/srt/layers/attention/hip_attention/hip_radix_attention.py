from __future__ import annotations

import os

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.hip_model_runner import HiPModelRunner
    from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInfo

from hip.models.hip_attention.gen3.attention_extend import (
    dual_stage_quadratic_hip_attention,
)
from hip.models.hip_attention.gen3.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
)
from hip.models.hip_attention.gen3.uvm_gpu_cache import HiPOffloadCache

logger = logging.getLogger(__name__)


_CHECKOUT_COUNTER = 0


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


class HiPRadixAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: HiPModelRunner):
        super().__init__()

        # NOTE: this backend instance is only one time creation.

        self.hip_config: HiPAttentionConfig = model_runner.hip_attention_config
        self.tp_rank = model_runner.tp_rank
        self.max_context_len = model_runner.model_config.context_len

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

        # logger.info(f'HiP attention is used in prompting (layer {layer.layer_id})!', stacklevel=0)

        is_offload_cache = isinstance(
            forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
        )

        if is_offload_cache:
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
            # offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None
        else:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            offload_cache = None

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        self.push_q_buffer(
            q_reshaped, layer_id=layer.layer_id, batch_size=forward_batch.batch_size
        )

        # Output tensor
        o = torch.empty_like(q_reshaped)

        start_len = 0
        decoding_reqs = []
        decoding_reqs_poistions = []
        for idx_batch, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_poistions.append(start_len)
            else:
                if is_offload_cache:
                    assert isinstance(
                        forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                    )
                    require_validation = (
                        forward_batch.token_to_kv_pool.require_validation
                    )
                    if require_validation:
                        k_chunk, v_chunk, k_pages, v_pages = (
                            forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                                layer_id=layer.layer_id,
                                batch_id=idx_batch,
                                cache_k=k[start_len : start_len + seq_len].unsqueeze(0),
                                cache_v=v[start_len : start_len + seq_len].unsqueeze(0),
                            )
                        )
                    else:
                        k_chunk, v_chunk = (
                            forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                                layer_id=layer.layer_id,
                                batch_id=idx_batch,
                                cache_k=k[start_len : start_len + seq_len].unsqueeze(0),
                                cache_v=v[start_len : start_len + seq_len].unsqueeze(0),
                            )
                        )
                    offload_cache = k_cache = v_cache = None
                else:
                    k_chunk = v_chunk = None

                if is_offload_cache:
                    # BUG: this padding is neccesary to match non offload scenario. why?
                    pad_size = self.max_context_len
                    if k_chunk.shape[1] != pad_size:
                        k_chunk_padded = torch.zeros(
                            (
                                k_chunk.shape[0],
                                pad_size,
                                k_chunk.shape[2],
                                k_chunk.shape[3],
                            ),
                            dtype=k_chunk.dtype,
                            device=k_chunk.device,
                        )
                        k_chunk_padded[:, : k_chunk.shape[1]] = k_chunk
                        del k_chunk
                        v_chunk_padded = torch.zeros(
                            (
                                v_chunk.shape[0],
                                pad_size,
                                v_chunk.shape[2],
                                v_chunk.shape[3],
                            ),
                            dtype=v_chunk.dtype,
                            device=v_chunk.device,
                        )
                        v_chunk_padded[:, : v_chunk.shape[1]] = v_chunk
                        del v_chunk
                        k_chunk = k_chunk_padded
                        v_chunk = v_chunk_padded

                    o_req, _ = self.forward_paged_hip(
                        query=q_reshaped[start_len : start_len + seq_len],
                        sm_scale=layer.scaling,
                        batch_size=1,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions[
                            start_len : start_len + seq_len
                        ],
                        seq_lens=forward_batch.seq_lens[idx_batch : idx_batch + 1],
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices[
                            idx_batch : idx_batch + 1
                        ],
                        layer=layer,
                        is_dense=layer.layer_id in self.hip_config.dense_layers,
                        k=k_chunk,
                        v=v_chunk,
                        online_update_cache=forward_batch.token_to_kv_pool.online_update_cache,
                        is_decode=False,
                    )

                    if require_validation:
                        o_req_valid, _ = self.forward_paged_hip(
                            query=q_reshaped[start_len : start_len + seq_len],
                            sm_scale=layer.scaling,
                            batch_size=1,
                            k_cache=k_pages,
                            v_cache=v_pages,
                            offload_cache=None,
                            positions=forward_batch.positions[
                                start_len : start_len + seq_len
                            ],
                            seq_lens=forward_batch.seq_lens[idx_batch : idx_batch + 1],
                            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                            req_pool_indices=forward_batch.req_pool_indices[
                                idx_batch : idx_batch + 1
                            ],
                            layer=layer,
                            is_dense=layer.layer_id in self.hip_config.dense_layers,
                            is_decode=False,
                        )

                        o_err = ((o_req - o_req_valid) ** 2).sum()
                        assert o_err < 1e-6, o_err

                    o[start_len : start_len + seq_len] = o_req
                else:
                    o_req, _ = self.forward_paged_hip(
                        query=q_reshaped[start_len : start_len + seq_len],
                        sm_scale=layer.scaling,
                        batch_size=1,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions[
                            start_len : start_len + seq_len
                        ],
                        seq_lens=forward_batch.seq_lens[idx_batch : idx_batch + 1],
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices[
                            idx_batch : idx_batch + 1
                        ],
                        layer=layer,
                        is_dense=layer.layer_id in self.hip_config.dense_layers,
                        k=k_chunk,
                        v=v_chunk,
                        is_decode=False,
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

        # logger.info(f'HiP attention is used in decoding (layer {layer.layer_id})!', stacklevel=0)

        is_offload_cache = isinstance(
            forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
        )

        metadata = None
        if forward_batch.hip_use_cached_mask or (
            forward_batch.hip_metadata_cached_stage is not None
        ):
            metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                layer.layer_id,
                q.shape[0],
                forward_batch.batch_size,
                forward_batch.hip_metadata_cached_stage,
            )

        require_validation = False
        if is_offload_cache:
            assert isinstance(
                forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
            )
            require_validation = forward_batch.token_to_kv_pool.require_validation

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                        async_copy=False,
                        push_to_gpu_cache=True,
                    )

            if not require_validation:
                k_cache = v_cache = None
                offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
            else:
                offload_cache, k_cache, v_cache = (
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                )
        else:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            offload_cache = None

        self.push_q_buffer(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            layer_id=layer.layer_id,
            batch_size=forward_batch.batch_size,
        )
        q_for_masking = self.get_q_buffer(layer.layer_id, forward_batch.batch_size)

        if not require_validation:
            o, metadata = self.forward_paged_hip(
                query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                query_for_mask=q_for_masking,
                sm_scale=layer.scaling,
                batch_size=forward_batch.batch_size,
                k_cache=k_cache,
                v_cache=v_cache,
                offload_cache=offload_cache,
                positions=forward_batch.positions,
                seq_lens=forward_batch.seq_lens,
                req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                layer=layer,
                cached_metadata=metadata,
                is_dense=layer.layer_id in self.hip_config.dense_layers,
                online_update_cache=(
                    forward_batch.token_to_kv_pool.online_update_cache
                    if isinstance(
                        forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                    )
                    else None
                ),
                is_decode=True,
            )
        else:

            def sse(a: torch.Tensor, b: torch.Tensor):
                assert a.dtype == b.dtype
                return ((a - b) ** 2).sum().item()

            err_k = sse(offload_cache.k_uvm.bank_gpu, k_cache)
            err_v = sse(offload_cache.v_uvm.bank_gpu, v_cache)

            o, metadata_new = self.forward_paged_hip(
                query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                query_for_mask=q_for_masking,
                sm_scale=layer.scaling,
                batch_size=forward_batch.batch_size,
                k_cache=None,
                v_cache=None,
                offload_cache=offload_cache,
                # NOTE: to test uvm only
                # k_cache=offload_cache.k_uvm.bank_gpu,
                # v_cache=offload_cache.v_uvm.bank_gpu,
                # offload_cache=None,
                # NOTE: to test on gpu only
                # k_cache=k_cache,
                # v_cache=v_cache,
                # offload_cache=None,
                positions=forward_batch.positions,
                seq_lens=forward_batch.seq_lens,
                req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                layer=layer,
                cached_metadata=metadata,
                is_dense=layer.layer_id in self.hip_config.dense_layers,
                online_update_cache=(
                    forward_batch.token_to_kv_pool.online_update_cache
                    if isinstance(
                        forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                    )
                    else None
                ),
                is_decode=True,
            )

            o_valid, metadata_valid = self.forward_paged_hip(
                query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                query_for_mask=q_for_masking,
                sm_scale=layer.scaling,
                batch_size=forward_batch.batch_size,
                k_cache=k_cache,
                v_cache=v_cache,
                offload_cache=None,
                positions=forward_batch.positions,
                seq_lens=forward_batch.seq_lens,
                req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                layer=layer,
                cached_metadata=metadata,
                is_dense=layer.layer_id in self.hip_config.dense_layers,
                online_update_cache=(
                    forward_batch.token_to_kv_pool.online_update_cache
                    if isinstance(
                        forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                    )
                    else None
                ),
                is_decode=True,
            )

            err_thresh = 1e-7

            o_sse = sse(o, o_valid)
            err_retry = -1
            err_uvm = None
            if o_sse >= err_thresh:
                indices_err = sse(metadata_new.indices, metadata_valid.indices)
                ks_err = sse(metadata_new.ks, metadata_valid.ks)
                ks_count_err = sse(metadata_new.ks_count, metadata_valid.ks_count)
                ks_start_end_err = sse(
                    metadata_new.ks_start_end, metadata_valid.ks_start_end
                )
                if (metadata_valid.stage_caches is not None) and (
                    len(metadata_valid.stage_caches) > 0
                ):
                    stage1_left_err = sse(
                        metadata_new.stage_caches[1].indices_left,
                        metadata_valid.stage_caches[1].indices_left,
                    )
                    stage1_right_err = sse(
                        metadata_new.stage_caches[1].indices_right,
                        metadata_valid.stage_caches[1].indices_right,
                    )
                    stage1_score_err = sse(
                        metadata_new.stage_caches[1].out_scores,
                        metadata_valid.stage_caches[1].out_scores,
                    )
                    stage2_left_err = sse(
                        metadata_new.stage_caches[2].indices_left,
                        metadata_valid.stage_caches[2].indices_left,
                    )
                    stage2_right_err = sse(
                        metadata_new.stage_caches[2].indices_right,
                        metadata_valid.stage_caches[2].indices_right,
                    )
                    stage2_score_err = sse(
                        metadata_new.stage_caches[2].out_scores,
                        metadata_valid.stage_caches[2].out_scores,
                    )
                else:
                    stage1_left_err = stage1_right_err = stage1_score_err = (
                        stage2_left_err
                    ) = stage2_right_err = stage2_score_err = None
                online_update = (
                    forward_batch.token_to_kv_pool.online_update_cache
                    if isinstance(
                        forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                    )
                    else None
                )

                o_uvm, metadata_uvm = self.forward_paged_hip(
                    query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    query_for_mask=q_for_masking,
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k_cache=offload_cache.k_uvm.bank_gpu,
                    v_cache=offload_cache.v_uvm.bank_gpu,
                    offload_cache=None,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    layer=layer,
                    cached_metadata=metadata,
                    is_dense=layer.layer_id in self.hip_config.dense_layers,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.online_update_cache
                        if isinstance(
                            forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                        )
                        else None
                    ),
                    is_decode=True,
                )

                offload_cache.sa_kv_cache.flush()
                offload_cache.mask_k_cache.flush()

                o_retry, metadata_retry = self.forward_paged_hip(
                    query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    query_for_mask=q_for_masking,
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k_cache=None,
                    v_cache=None,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    layer=layer,
                    cached_metadata=metadata,
                    is_dense=layer.layer_id in self.hip_config.dense_layers,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.online_update_cache
                        if isinstance(
                            forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                        )
                        else None
                    ),
                    is_decode=True,
                )
                err_uvm = sse(o, o_uvm)
                err_retry = sse(o_valid, o_retry)

                print(o)
                print(o_valid)
                print(metadata_new.indices)
                print(metadata_valid.indices)

                assert (
                    o_sse < err_thresh
                ), f"""
sse={o_sse}
err_k (uvm_k <=> valid_k) = {err_k}
err_v (uvm_v <=> valid_v) ={err_v}
err_retry (o_valid <=> o_retry) = {err_retry}
err_uvm (o_first <=> o_uvm_retry) = {err_uvm}
indices_err={indices_err}
ks_err={ks_err}
ks_count_err={ks_count_err}
ks_start_end_err={ks_start_end_err}
stage1_left_err={stage1_left_err}
stage1_right_err={stage1_right_err}
stage1_score_err={stage1_score_err}
stage2_left_err={stage2_left_err}
stage2_right_err={stage2_right_err}
stage2_score_err={stage2_score_err}
online_update={online_update}
"""

            metadata = metadata_new

        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
            layer_id=layer.layer_id,
            size=q.shape[0],
            batch_size=forward_batch.batch_size,
            metadata=metadata,
        )

        if is_offload_cache:
            offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_paged_hip(
        self,
        query: torch.Tensor,
        sm_scale: float,
        batch_size: int,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        offload_cache: Optional[HiPOffloadCache],
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_tokens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        layer: RadixAttention,
        cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
        is_dense: bool = False,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        query_for_mask: Optional[torch.Tensor] = None,
        online_update_cache: bool = False,
        is_decode: bool = False,
    ) -> tuple[torch.Tensor, "HiPAttentionOutputMetadata"]:
        global _CHECKOUT_COUNTER
        N, num_heads, hidden_dims = query.shape
        dst_seq_len = N // batch_size

        is_dense = layer.layer_id in self.hip_config.dense_layers
        if not is_decode:
            if len(self.hip_config.prefill_layers) == 2:
                layer_config = self.hip_config.prefill_layers[0 if is_dense else 1]
            else:
                layer_config = self.hip_config.prefill_layers[layer.layer_id]
        else:
            assert dst_seq_len == 1
            if len(self.hip_config.layers) == 2:
                layer_config = self.hip_config.layers[0 if is_dense else 1]
            else:
                layer_config = self.hip_config.layers[layer.layer_id]

        query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)
        if query_for_mask is not None:
            query_for_mask = query_for_mask.view(batch_size, -1, num_heads, hidden_dims)

        if k_cache is not None:
            N_PAGE, num_heads_kv, hidden_dims_kv = k_cache.shape
            assert v_cache.shape == k_cache.shape
            assert hidden_dims_kv == hidden_dims

            k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)
            v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)

        # FIXME: this operation is linear during decoding
        block_table = req_to_tokens.index_select(dim=0, index=req_pool_indices)

        BLOCK_TABLE_BSZ, MODEL_SEQ_LEN = block_table.shape
        assert batch_size == BLOCK_TABLE_BSZ

        # NOTE(heejun): the whole point to need to find gemma is large size of hidden size
        # FIXME: find better way to detect Gemma
        if k_cache is not None:
            hidden_size = k_cache.shape[-1]
        elif k is not None:
            hidden_size = k.shape[-1]
        elif offload_cache is not None:
            hidden_size = offload_cache.k_uvm.bank_cpu.shape[-1]
        else:
            raise Exception()
        is_gemma = hidden_size > 128

        require_cache_statistics = False
        if cached_metadata is None:
            require_cache_statistics = True
        elif cached_metadata.indices is None:
            require_cache_statistics = True
        elif os.getenv("HIP_DISABLE_COMPUTE_STATISTICS", "1") == "0":
            require_cache_statistics = True

        if query_for_mask is not None:
            query_position_ids = positions.view(batch_size, dst_seq_len)
            position_ids = (
                torch.arange(0, query_for_mask.shape[1], device=query.device)[None, :]
                - (query_for_mask.shape[1] - 1)
                + query_position_ids
            )
        else:
            position_ids = positions.view(batch_size, dst_seq_len)

        args = HiPAttentionArgs(
            k_cache=(
                k_cache.view(torch.uint8)
                if isinstance(k_cache, torch.Tensor)
                and k_cache.dtype == torch.float8_e5m2
                else k_cache
            ),
            v_cache=(
                v_cache.view(torch.uint8)
                if isinstance(k_cache, torch.Tensor)
                and v_cache.dtype == torch.float8_e5m2
                else v_cache
            ),
            offload_cache=offload_cache,
            block_table=block_table,
            cache_seq_lens=seq_lens,
            position_ids=position_ids,
            block_size_k=32 if is_gemma else 64,  # BLOCK_CHUNK
            sliding_window_size=layer_config.sliding_window_size,
            sink_token_size=layer_config.sink_token_size,
            using_extend=self.hip_config.using_extend,
            need_apply_rope=self.hip_config.using_extend,
            rope_cos=layer.rope_cos,
            rope_sin=layer.rope_sin,
            logit_softcap=layer.logit_cap if layer.logit_cap != 0.0 else None,
            second_stage_k=layer_config.second_stage_k,
            stages=layer_config.stages,
            model_context_length=layer.orig_context_len,
            extend_context_length=self.max_context_len,
            block_sparse_block_size_q=self.hip_config.block_sparse_block_size_q,
            scan_extend_backend=(
                (
                    "relative"
                    if self.hip_config.apply_v_dot
                    else ("streaming" if is_dense else "relative")
                )
                if layer_config.scan_extend_backend is None
                else layer_config.scan_extend_backend
            ),
            sa_extend_backend=layer_config.sa_extend_backend,
            online_update_cache=online_update_cache,
            require_cache_statistics=require_cache_statistics,
            disable_flashdecode=not is_decode,
            q_mask=(
                (query_for_mask * sm_scale).to(query.dtype)
                if query_for_mask is not None
                else None
            ),
            sliding_window_indices=(
                self.diag_sliding_window_indices[layer.layer_id]
                if self.diag_sliding_window_indices is not None
                else None
            ),
            _layer_id=layer.layer_id,
        )

        last_dense = 64

        if is_decode or (query.shape[1] < (last_dense * 2)) or (last_dense <= 0):
            context, metadata = dual_stage_quadratic_hip_attention(
                (query * sm_scale).to(query.dtype),
                k,
                v,
                args=args,
                cached_metadata=cached_metadata,
            )
            context = context.to(query.dtype)
            context = context[:, -query.shape[1] :, :, :].contiguous()
        else:
            if layer.layer_id < 160:
                assert query_for_mask is None
                position_ids = args.position_ids
                args.position_ids = position_ids[:, :-last_dense]
                context, metadata = dual_stage_quadratic_hip_attention(
                    (query[:, :-last_dense, :, :] * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args,
                    cached_metadata=cached_metadata,
                )
                context_sparse = context.to(query.dtype)

                args.sliding_window_size = 777
                args.position_ids = position_ids[:, -last_dense:]
                context, metadata = dual_stage_quadratic_hip_attention(
                    (query[:, -last_dense:, :, :] * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args,
                    cached_metadata=cached_metadata,
                )
                context_dense = context.to(query.dtype)

                context = torch.cat([context_sparse, context_dense], dim=1)
            else:
                assert query_for_mask is None
                block_size_q = args.stages[-1].stage_block_size_q
                k_bos = args.k_cache[args.block_table[:, :1], 0, :, :]
                k_bos = k_bos / k_bos.float().square().sum(dim=-1, keepdim=True).sqrt()
                q_norm = query / query.float().square().sum(dim=-1, keepdim=True).sqrt()
                # T_q
                scores = torch.matmul(
                    q_norm.permute(0, 2, 1, 3),
                    k_bos.permute(0, 2, 3, 1).repeat_interleave(
                        q_norm.shape[2] // k_bos.shape[2], 1
                    ),
                )[0, :, :, 0].mean(dim=0)

                # scores = -torch.arange(0, scores.shape[0], device=scores.device, dtype=scores.dtype)

                # print(scores)
                half_window = 17
                scores = scores[None, None, :]
                # scores = torch.nn.functional.pad(scores[None, None, :], (half_window, half_window), mode='replicate')
                # scores = torch.nn.functional.avg_pool1d(
                #     scores, kernel_size=half_window*2+1, stride=1, padding=0
                # )
                # print(scores)
                scores = torch.nn.functional.pad(
                    scores,
                    (
                        0,
                        (
                            block_size_q - (scores.shape[-1] % block_size_q)
                            if scores.shape[-1] % block_size_q
                            else 0
                        ),
                    ),
                    mode="replicate",
                )
                scores = -torch.nn.functional.max_pool1d(
                    -scores, kernel_size=block_size_q, stride=block_size_q, padding=0
                )[0, 0, :]
                # print(scores)
                scores[-4:].fill_(float("-inf"))
                # print(scores)
                scores = scores.repeat_interleave(block_size_q, 0)
                scores = scores[: q_norm.shape[1]]
                num_dense = 1024  # int(scores.shape[-1] * 0.025)
                # print(num_dense)
                num_dense = (
                    num_dense
                    + block_size_q
                    - (
                        (num_dense % block_size_q)
                        if num_dense % block_size_q
                        else block_size_q
                    )
                )
                # print(2, num_dense)
                num_dense = num_dense + q_norm.shape[1] % block_size_q
                # print(3, num_dense)
                num_dense = num_dense + block_size_q
                # print(4, num_dense)
                num_dense = max(64 + q_norm.shape[1] % block_size_q, num_dense)
                # print(5, num_dense)
                # num_dense = 256
                # print(num_dense, q_norm.shape[1] % block_size_q)
                _, dense_indices = torch.topk(
                    -scores, dim=-1, k=num_dense, largest=True, sorted=True
                )
                # print(scores, scores.shape, num_dense)
                # print(dense_indices)
                dense_indices = dense_indices.sort().values
                # dense_indices = scores.shape[-1] - dense_indices - 1
                # print('a', dense_indices)
                # dense_indices = dense_indices // block_size_q * block_size_q
                # dense_indices = (dense_indices[::block_size_q, None] + torch.arange(0, block_size_q, device=query.device)[None, :]).view(-1)[:dense_indices.shape[-1]]
                # dense_indices = scores.shape[-1] - dense_indices - 1
                print("b", dense_indices[::block_size_q], query.shape)
                sparse_indices = torch.arange(0, scores.shape[-1], device=query.device)
                sparse_indices.scatter_(dim=0, index=dense_indices, value=987654321)
                sparse_indices, _ = sparse_indices.sort()
                sparse_indices = sparse_indices[:-num_dense]

                check = torch.zeros((scores.shape[-1],), device=query.device)
                check.scatter_(dim=0, index=sparse_indices, value=-1)
                check.scatter_(dim=0, index=dense_indices, value=1)
                check = (check == 0).nonzero()
                # print((check == 0).nonzero(), query.shape[1], scores.shape, dense_indices, flush=True)
                assert check.shape[0] == 0, check
                assert (query.shape[1] - 1) in dense_indices
                check = ((dense_indices[::block_size_q] % block_size_q) != 0).nonzero()
                assert check.shape[0] == 0, check
                # assert ((query.shape[1] - 64) in dense_indices)

                dense_queries = query[:, dense_indices, :, :]
                sparse_queries = query[:, sparse_indices, :, :]

                position_ids = args.position_ids
                dense_pos_ids = position_ids[:, dense_indices]
                sparse_pos_ids = position_ids[:, sparse_indices]

                args.q_mask = None
                # args.sliding_window_size = 777  # NOTE: this 777 is correct
                args.position_ids = sparse_pos_ids
                context, metadata = dual_stage_quadratic_hip_attention(
                    (sparse_queries * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args,
                    cached_metadata=cached_metadata,
                )
                context_sparse = context.to(query.dtype)

                args.sliding_window_size = 777  # NOTE: this 777 is correct
                args.position_ids = dense_pos_ids
                context, metadata = dual_stage_quadratic_hip_attention(
                    (dense_queries * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args,
                    cached_metadata=cached_metadata,
                )
                context_dense = context.to(query.dtype)

                context = torch.full_like(query, fill_value=42)
                # context = context_all.to(query.dtype).clone()
                context.scatter_(
                    dim=1,
                    index=dense_indices[None, :, None, None].expand_as(context_dense),
                    src=context_dense,
                )
                context.scatter_(
                    dim=1,
                    index=sparse_indices[None, :, None, None].expand_as(context_sparse),
                    src=context_sparse,
                )

                check = (context == 42).nonzero()
                assert check.shape[0] == 0, f"{check} {check.shape}"
                # print(context)

        layers_to_capture = [0, 1, 2, 3, 4, 8, 12, 16, 24, 31]
        NEED_CHECKOUT = os.getenv("HIP_DEBUG_NEED_CHECKOUT", "0") == "1"
        if (
            NEED_CHECKOUT
            and (self.tp_rank == 0)
            and is_decode
            and (layer.layer_id in layers_to_capture)
        ):
            root = "./saves/sglang_decode"
            if not os.path.exists(root):
                _CHECKOUT_COUNTER = 0
            filename = (
                f"{root}/checkout_sample_{_CHECKOUT_COUNTER}_layer_{layer.layer_id}.pth"
            )
            os.makedirs(root, exist_ok=True)
            torch.save(
                {
                    "q": query,
                    "sm_scale": sm_scale,
                    "k": (
                        k
                        if k is not None
                        else args.gather_k_from_paged_cache(chunk_size=1)
                    ),
                    "v": (
                        v
                        if k is not None
                        else args.gather_v_from_paged_cache(chunk_size=1)
                    ),
                    "block_table": block_table,
                    "cos": layer.rope_cos,
                    "sin": layer.rope_sin,
                    "out": context,
                    "metadata": metadata,
                },
                filename,
            )
            print(f"saved {filename}")
            if layer.layer_id == max(layers_to_capture):
                _CHECKOUT_COUNTER += 1

        return context.view(N, num_heads, hidden_dims), metadata
