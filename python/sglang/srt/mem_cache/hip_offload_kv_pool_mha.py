import time
import torch
from torch import Tensor
from typing import Dict, Optional, Set, Tuple, Union
import threading

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool
)

from hip.models.hip_attention.gen3.uvm_gpu_cache import (
    UVMCache,
    GPUCache,
    HiPOffloadCache,
    format_size_bytes,
)
import logging

logger = logging.getLogger(__name__)

class MHATokenToHiPOffloadKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device,
    ):
        assert isinstance(device, torch.device)
        assert device.index is not None
        
        super().__init__(size, dtype, device)
        
        #TODO: derive token sizes from size
        self.head_num = head_num
        self.head_dim = head_dim
        self.max_mask_cache_token_size = 32 * 1024
        self.max_sa_cache_token_size = 4 * 1024
        
        self.layer_buffer = [
            HiPOffloadCache(
                max_token_size=size + 1,
                max_mask_cache_token_size=self.max_mask_cache_token_size,
                max_sa_cache_token_size=self.max_sa_cache_token_size,
                head_num=head_num,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        
        # (layer_id, batch_id) -> (K, V, seq_len)
        self.prefetch_threads: Dict[Tuple[int, int], threading.Thread] = {}
        self.prefetched_kv: Dict[Tuple[int, int], Tuple[Tensor, Tensor, int]] = {}
        
        self.async_set_threads: Set[threading.Thread] = set()
        
        uvm_allocated_bytes = 0
        gpu_allocated_bytes = 0
        for cache in self.layer_buffer:
            uvm_allocated_bytes += cache.k_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.k_uvm.allocated_gpu_bytes
            uvm_allocated_bytes += cache.v_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.v_uvm.allocated_gpu_bytes
            gpu_allocated_bytes += cache.mask_k_cache.allocated_gpu_bytes
            gpu_allocated_bytes += cache.sa_kv_cache.allocated_gpu_bytes
        logger.info(
            f'Allocated CPU(UVM) bytes: {format_size_bytes(uvm_allocated_bytes)}, '
            f'Allocated GPU bytes: {format_size_bytes(gpu_allocated_bytes)}'
        )

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> HiPOffloadCache:
        # Use this function for decode, pass this to `k`
        return self.layer_buffer[layer_id]
    
    def prefetch_prefix_kv_buffer(
        self, 
        layer_id: int, 
        batch_id: int, 
        table: Tensor, 
        prefix_seq_len: int,
    ) -> threading.Thread:
        # you must call before get fetched prefix
        assert table.ndim == 1
        
        hip_offload_cache = self.get_kv_buffer(layer_id)
        
        handle_id = (layer_id, batch_id)
        # print(threading.current_thread().native_id, 'start prefetch')
        # start_event = torch.cuda.Event()
        def thread_main():
            torch.cuda.synchronize(device=self.device)
            # print(threading.current_thread().native_id, 'start copy')
            stream = torch.cuda.Stream(device=self.device, priority=0)
            stream.wait_stream(torch.cuda.default_stream(device=self.device))
            # t_local = time.time()
            with torch.cuda.stream(stream):
                # start_event.synchronize()
                k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                    table=table,
                    device=self.device,
                )
                assert k.device == self.device
                assert v.device == self.device
                self.prefetched_kv[handle_id] = (k, v, prefix_seq_len)
            stream.synchronize()
            self.prefetch_threads.pop(handle_id)
            # print(threading.current_thread().native_id, 'done copy', (time.time() - t_local) * 1000, (k.numel() * k.element_size()) * 2 / 1024 / 1024)
        t = threading.Thread(target=thread_main, daemon=True)
        self.prefetch_threads[handle_id] = t
        t.start()
        
        return t
    
    def get_fetched_prefix_kv_buffer(
        self, 
        layer_id: int,
        batch_id: int,
        # you need to pass KV for extend
        cache_k: Tensor,
        cache_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Use this function for prefill
        handle_id = (layer_id, batch_id)
        prefetch_thread = self.prefetch_threads.get(handle_id, None)
        if prefetch_thread is not None:
            prefetch_thread.join()
        
        assert handle_id in self.prefetched_kv, "did prefetch successed?"
        k, v, seq_len = self.prefetched_kv.pop(handle_id)
        
        assert isinstance(k, Tensor)
        assert isinstance(v, Tensor)
        assert isinstance(seq_len, int)
        assert k.shape == v.shape
        assert k.ndim == 4, f'{k.shape}'
        assert k.shape[0] == 1
        assert k.shape[1] >= seq_len
        assert k.shape[2] == self.head_num
        assert k.shape[3] == self.head_dim
        assert k.dtype == v.dtype
        assert k.dtype == self.dtype
        assert k.shape[1] >= seq_len+cache_k.shape[1]
        assert cache_k.ndim == 4
        assert cache_k.shape[0] == 1
        assert cache_k.shape[2] == self.head_num
        assert cache_k.shape[3] == self.head_dim
        
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype, non_blocking=True)
            cache_v = cache_v.to(self.dtype, non_blocking=True)
        
        k[:, seq_len:seq_len+cache_k.shape[1], :, :].copy_(cache_k, non_blocking=True)
        v[:, seq_len:seq_len+cache_v.shape[1], :, :].copy_(cache_v, non_blocking=True)
        
        return k, v

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        async_copy: bool = False,
    ):
        layer_id = layer.layer_id
        # pass async_copy=True when only prefill (eager mode)
        assert (not async_copy) or (async_copy and (not torch.cuda.is_current_stream_capturing()))
        
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        
        if async_copy:
            # start_event = torch.cuda.Event()
            def thread_main():
                torch.cuda.synchronize(device=self.device)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.default_stream(device=self.device))
                with torch.cuda.stream(stream):
                    # start_event.synchronize()
                    table_gpu = table
                    table_cpu = table.to('cpu', non_blocking=False)
                    cache_k_cpu = cache_k.to('cpu', non_blocking=False)
                    cache_v_cpu = cache_v.to('cpu', non_blocking=False)
                    self.layer_buffer[layer_id].set_kv_buffer(
                        table=table_cpu,
                        table_gpu=table_gpu,
                        cache_k=cache_k_cpu,
                        cache_v=cache_v_cpu,
                    )
                stream.synchronize()
                self.async_set_threads.remove(t)
            t = threading.Thread(target=thread_main, daemon=True)
            self.async_set_threads.add(t)
            t.start()
        else:
            self.layer_buffer[layer_id].set_kv_buffer(
                table=table,
                table_gpu=table,
                cache_k=cache_k,
                cache_v=cache_v,
            )
    
    def synchronize(self):
        torch.cuda.synchronize(device=self.device)
        t = time.time()
        # you must call this function when finish prefill, before decode
        while (len(self.prefetch_threads) > 0) or (len(self.async_set_threads) > 0):
            time.sleep(0.001)
        assert len(self.prefetch_threads) == 0
        assert len(self.async_set_threads) == 0
        assert len(self.prefetched_kv) == 0
        elapsed = time.time() - t
        logger.debug(f'Final layer sync took {elapsed * 1024:.4f} ms')