from transformers import CLIPProcessor, CLIPModel
import os

from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
import decord
import asyncio
import torch

os.environ["NFRAMES"] = "64"

async def test():
    vr = decord.VideoReader("../86CxyhFV9MI.mp4", ctx=decord.gpu(0))
    vr_cpu = decord.VideoReader("../86CxyhFV9MI.mp4", ctx=decord.cpu(0))
    vr = (vr, vr_cpu)
    aks_model_card = "openai/clip-vit-base-patch32"

    kwargs = dict(
        aks_clip_processor=CLIPProcessor.from_pretrained(aks_model_card, use_fast=True),
        aks_clip_model=CLIPModel.from_pretrained(aks_model_card, device_map={
            '': 0
        }).eval()
    )

    await preprocess_video(vr, "aks", **kwargs)

    torch.cuda.set_sync_debug_mode("warn")
    await preprocess_video(vr, "aks", **kwargs)
    torch.cuda.set_sync_debug_mode("default")

    await preprocess_video(vr, "aks", **kwargs)


asyncio.run(test())
