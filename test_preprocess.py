import os
from transformers import CLIPProcessor, CLIPModel
import decord
import asyncio
import torch

from byteclip.inference.segment_selection.byteclip.visual_utils import H264Video
from byteclip.models.modeling_modernbytebert import ModernByteBertModelWithProjection
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video

os.environ["NFRAMES"] = "64"

async def test():
    video_file_name = "../86CxyhFV9MI_720p.mp4"
    # video_file_name = "../UO_6TQnnOxM.mp4"
    vr = decord.VideoReader(video_file_name, ctx=decord.gpu(0))
    vr_cpu = decord.VideoReader(video_file_name, ctx=decord.cpu(0))
    h264_video = H264Video(video_file_name)
    vr = (vr, vr_cpu, h264_video)

    frame_selection_method = "byteclip"
    aks_model_card = "openai/clip-vit-base-patch32"
    byteclip_config_path = "/home/park/devel/byteCLIP/src/byteclip/inference/segment_selection/byteclip/sample_model_config.json"

    kwargs = dict(
        clip_processor=CLIPProcessor.from_pretrained(aks_model_card, use_fast=True),
        clip_model=CLIPModel.from_pretrained(aks_model_card, device_map={
            '': 0
        }).eval(),
        byteclip_model=ModernByteBertModelWithProjection.from_config(byteclip_config_path).to('cuda:0'),
    )

    await preprocess_video(vr, frame_selection_method, **kwargs)

    torch.cuda.set_sync_debug_mode("warn")
    await preprocess_video(vr, frame_selection_method, **kwargs)
    torch.cuda.set_sync_debug_mode("default")

    await preprocess_video(vr, frame_selection_method, **kwargs)


asyncio.run(test())
