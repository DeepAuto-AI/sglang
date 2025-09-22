#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serve /generate with HF + MInference (multi-GPU via device_map sharding; NOT real TP).

POST /generate
{
  "text": "prompt" | ["p1","p2",...],
  "sampling_params": {
      "max_new_tokens": 128,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "stop": ["</end>"]
  },
  "stream": false
}
Returns {"text": "..."} or [{"text":"..."},...].
"""

import argparse
import json
import os
import sys

# --- MInference patcher
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from minference import MInference
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# --- HF deps
from transformers.cache_utils import DynamicCache

# Optional: presence/frequency penalties (newer Transformers)
try:
    from transformers.generation.logits_process import (
        FrequencyPenaltyLogitsProcessor,
        PresencePenaltyLogitsProcessor,
    )

    _HAS_P_F = True
except Exception:
    _HAS_P_F = False

try:
    from accelerate import infer_auto_device_map  # noqa: F401 (just to detect)

    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False


# ==================== Helpers ====================


class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_ids: List[List[int]]):
        super().__init__()
        self.stop_ids = [torch.tensor(s, dtype=torch.long) for s in stop_ids if len(s)]
        self.maxL = max((len(s) for s in self.stop_ids), default=0)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.maxL == 0:
            return False
        cur = input_ids[0] if input_ids.dim() == 2 else input_ids
        if cur.size(0) < self.maxL:
            return False
        tail = cur[-self.maxL :]
        for s in self.stop_ids:
            L = s.size(0)
            if L and L <= tail.size(0):
                if torch.equal(tail[-L:], s):
                    return True
        return False


def _map_dtype(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    if s == "auto":
        return "auto"
    # default
    return torch.bfloat16


def _eos_id_list(tok: AutoTokenizer) -> List[int]:
    ids = []
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        if isinstance(tok.eos_token_id, int):
            ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            ids.extend([i for i in tok.eos_token_id if isinstance(i, int)])
    return list(dict.fromkeys(ids))


def _map_sampling(params: Dict[str, Any]) -> Dict[str, Any]:
    max_new = int(params.get("max_new_tokens", 128))
    temperature = float(params.get("temperature", 0.0))
    do_sample = params.get("do_sample")
    if do_sample is None:
        do_sample = temperature > 0.0
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 0))
    rep_pen = float(params.get("repetition_penalty", 1.0))
    pres_pen = float(params.get("presence_penalty", 0.0))
    freq_pen = float(params.get("frequency_penalty", 0.0))
    stop = params.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    return dict(
        max_new_tokens=max_new,
        temperature=temperature if do_sample else 1.0,
        do_sample=bool(do_sample),
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_pen,
        presence_penalty=pres_pen,
        frequency_penalty=freq_pen,
        stop=stop,
    )


def _get_max_ctx(model, tokenizer) -> int:
    # Conservative: prefer model.config.max_position_embeddings if sane; else tokenizer.model_max_length.
    m = getattr(model.config, "max_position_embeddings", None)
    if isinstance(m, int) and m > 0 and m < 10**9:
        return m
    t = getattr(tokenizer, "model_max_length", None)
    if isinstance(t, int) and t > 0 and t < 10**9:
        return t
    # Fallback large number if unknown
    return 2**31 - 1


def _prune_middle(
    input_ids: torch.LongTensor, max_ctx: int, max_new: int
) -> Tuple[torch.LongTensor, int]:
    """
    If len(input_ids) + max_new > max_ctx, remove exactly 'excess' tokens from the middle.
    Returns (possibly pruned ids, num_pruned).
    """
    L = input_ids.size(1)
    overflow = L + max_new - max_ctx
    if overflow <= 0:
        return input_ids, 0
    # Compute symmetric middle cut
    start = max(
        1, (L // 2) - (overflow // 2)
    )  # keep at least the very first token (pos enc anchor)
    end = min(L - 1, start + overflow)  # keep the very last token
    if end <= start:
        # pathological tiny prompts; just drop from the middle index
        mid = L // 2
        sl = min(overflow, max(0, L - 2))
        start = max(1, mid - sl // 2)
        end = min(L - 1, start + sl)
    keep_left = input_ids[:, :start]
    keep_right = input_ids[:, end:]
    pruned = torch.cat([keep_left, keep_right], dim=1)
    return pruned, overflow


def _tokenize_and_prune_batch(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_ctx: int,
    max_new: int,
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor, List[int]]:
    """
    Tokenize each prompt separately (no truncation), prune from the middle per prompt if needed,
    then left-pad to a batch.
    Returns (input_ids, attention_mask, pruned_counts)
    """
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.padding_side = "left"

    ids_list: List[torch.LongTensor] = []
    pruned_counts: List[int] = []

    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"]  # [1, L]
        ids, pruned = _prune_middle(ids, max_ctx, max_new)
        ids_list.append(ids)
        pruned_counts.append(pruned)

    # Pad to batch (left)
    maxL = max(x.size(1) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    batch = torch.full((len(ids_list), maxL), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), maxL), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        L = ids.size(1)
        batch[i, -L:] = ids[0]
        attn[i, -L:] = 1

    return batch.to(device), attn.to(device), pruned_counts


def _build_stopping(
    stop_strs: List[str], tokenizer: AutoTokenizer
) -> Optional[StoppingCriteriaList]:
    if not stop_strs:
        return None
    stop_ids = []
    for s in stop_strs:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            stop_ids.append(toks)
    if not stop_ids:
        return None
    return StoppingCriteriaList([StopOnSequences(stop_ids)])


def _build_logits_processors(pres_pen: float, freq_pen: float):
    lp = []
    if _HAS_P_F:
        if abs(pres_pen) > 1e-8:
            lp.append(PresencePenaltyLogitsProcessor(pres_pen))
        if abs(freq_pen) > 1e-8:
            lp.append(FrequencyPenaltyLogitsProcessor(freq_pen))
    # If not available, silently skip (HF older version)
    return lp if lp else None


# ==================== App / Global State ====================

app = FastAPI(title="HF + MInference /generate", version="1.0")

STATE: Dict[str, Any] = {
    "tokenizer": None,  # HF tokenizer
    "model": None,  # HF model
    "device": None,  # torch.device for input_ids
    "max_ctx": None,  # int
}


def load_hf_minf(
    model_id: str,
    dtype: str,
    attn_impl: str,
    trust_remote_code: bool,
):

    torch_dtype = _map_dtype(dtype)

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )

    load_kwargs = {
        "attn_implementation": attn_impl,
        "torch_dtype": torch_dtype if torch_dtype != "auto" else None,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto" if _HAS_ACCELERATE else None,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id, **{k: v for k, v in load_kwargs.items() if v is not None}
    )
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model = model.cuda()
    model.eval()

    # Patch with MInference (HF path). Try common signatures.
    patch = MInference("minference", model_name=model_id)
    model = patch(model)  # may modify in place or return wrapped model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Even with device_map="auto", inputs should be placed on the first CUDA device for generate().
    # (HF will scatter as needed.)
    first_cuda = torch.device("cuda:0")
    max_ctx = _get_max_ctx(model, tok)

    STATE.update(tokenizer=tok, model=model, device=first_cuda, max_ctx=max_ctx)


def hf_generate(prompts: List[str], sp: Dict[str, Any]) -> List[str]:
    model = STATE["model"]
    tok: AutoTokenizer = STATE["tokenizer"]
    device: torch.device = STATE["device"]
    max_ctx: int = STATE["max_ctx"]

    max_new = int(sp["max_new_tokens"])

    # Batch tokenize + prune
    input_ids, attention_mask, _ = _tokenize_and_prune_batch(
        tok, prompts, max_ctx, max_new, device
    )

    # Stopping criteria from stop strings (token-wise)
    stopping = _build_stopping(sp.get("stop", []), tok)

    # Optional presence/frequency penalties
    logits_proc = _build_logits_processors(
        sp.get("presence_penalty", 0.0), sp.get("frequency_penalty", 0.0)
    )

    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=bool(sp["do_sample"]),
        temperature=float(sp["temperature"]) if sp["do_sample"] else 0.0,
        use_cache=True,
        past_key_values=DynamicCache(),
        top_p=float(sp["top_p"]),
        repetition_penalty=float(sp.get("repetition_penalty", 1.0)),
        eos_token_id=_eos_id_list(tok),
        pad_token_id=tok.pad_token_id,
    )
    if int(sp.get("top_k", 0)) > 0:
        gen_kwargs["top_k"] = int(sp["top_k"])
    if stopping is not None:
        gen_kwargs["stopping_criteria"] = stopping
    if logits_proc is not None:
        gen_kwargs["logits_processor"] = logits_proc

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # Slice off the prompt part and decode
    outputs = []
    for i in range(out.size(0)):
        gen_only = out[i, input_ids.size(1) :]
        text = tok.decode(gen_only, skip_special_tokens=True)
        outputs.append(text)
    return outputs


# ==================== HTTP ====================


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.api_route("/generate", methods=["POST", "PUT"])
async def generate(req: Request):
    # --- parse body robustly (JSON string, dict, list, or form) ---
    try:
        raw = await req.body()
        body = None
        if raw and raw.strip():
            body = json.loads(raw)
        else:
            # Some clients send application/x-www-form-urlencoded with a "data" field
            try:
                form = await req.form()
                payload = form.get("data") or form.get("request")
                if payload:
                    body = json.loads(payload)
            except Exception:
                pass
        if body is None:
            raise ValueError("empty body")
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    # Allow top-level list as shorthand for {"text": [...]}
    if isinstance(body, list):
        body = {"text": body}
    if not isinstance(body, dict):
        raise HTTPException(400, "Body must be an object")

    # --- normalize fields RULER/OpenAI-like clients might use ---
    text = None
    for key in ("text", "prompt", "prompts", "input", "inputs"):
        if key in body:
            text = body[key]
            break
    if text is None:
        raise HTTPException(400, "Missing 'text'/'prompt'/'prompts'/'input'/'inputs'")

    # Always operate on a list of prompts
    prompts = text if isinstance(text, list) else [text]

    sp = _map_sampling(body.get("sampling_params", {}) or {})

    try:
        texts = hf_generate(prompts, sp)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Generation error: {e}")

    res = [{"text": t} for t in texts]
    return JSONResponse(res if isinstance(text, list) else res[0])


# ==================== CLI ====================


def parse_args():
    ap = argparse.ArgumentParser(description="HF + MInference /generate server")

    # common
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--trust-remote-code", action="store_true")

    # HF-specific
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["auto", "bf16", "fp16", "float16", "float32", "fp32"],
    )
    ap.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
    )

    return ap.parse_args()


def main():
    args = parse_args()

    if not _HAS_ACCELERATE:
        print(
            "[warn] accelerate not detected; model will be ignored and model will load on a single device.",
            file=sys.stderr,
        )

    load_hf_minf(
        model_id=args.model,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
