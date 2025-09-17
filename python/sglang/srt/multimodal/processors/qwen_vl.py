import asyncio
import math
import os
import re
import time
from typing import List, Union

import numpy as np
import nvtx

import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.utils import logger

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    width, height = image.size
    min_pixels = MIN_PIXELS
    max_pixels = MAX_PIXELS
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    return image


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


async def resize_image_async(image):
    return resize_image(image)


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def aks_cand_frame_indices(vr, sampling_rate):
    fps = vr.get_avg_fps()
    step = max(int(round(fps / sampling_rate)), 1)
    num_samples = (len(vr) + step - 1) // step
    frames_indices = []
    current_index = 0
    for i in range(num_samples):
        if current_index >= len(vr):
            break
        frames_indices.append(current_index)
        current_index += 1
        if i < num_samples - 1:
            current_index += step - 1
    return frames_indices


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def cosine_similarity(a, b):
    similarity = a @ b.T / (torch.linalg.norm(a) * torch.linalg.norm(b))
    # Reduce dimensions that have size 1
    for i in range(len(similarity.shape)):
        if similarity.shape[i] == 1:
            similarity = similarity.squeeze(axis=i)
            break
    return similarity


# process video, qwen-specific
async def preprocess_video(
    vr,
    frame_selection_method: str,
    clip_processor: torch.nn.Module = None,
    clip_model: torch.nn.Module = None,
    byteclip_model: torch.nn.Module = None,
    image_factor: int = IMAGE_FACTOR,
    # vr: VideoReader, image_factor: int = IMAGE_FACTOR
) -> torch.Tensor:
    with (nvtx.annotate(message="preprocess_video", color="green")):
        vr, vr_cpu, h264 = vr

        ele = {}
        total_frames, video_fps = len(vr_cpu), vr_cpu.get_avg_fps()

        if os.environ.get("NFRAMES") is not None:
            nframes = int(os.environ.get("NFRAMES"))
        else:
            print("NFRAMES is not set; deciding dynamically based on fps and total_frames")
            nframes = smart_nframes({}, total_frames=total_frames, video_fps=video_fps)

        print(f"!!! Using nframes: {nframes}")
        question = "What is the visual content of the video?"  # TODO: replace with actual question

        device = "cuda:0"

        if frame_selection_method == "uniform":
            idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

        elif frame_selection_method == "aks":
            from lmms_eval.frame_selection_utils.pipelines.pipeline_utils \
                import select_visuals

            print("!!! AKS start " + "=" * 80)

            sampling_rate = float(os.environ.get("AKS_SAMPLING_RATE", "1.0"))
            batch_size = int(os.environ.get("AKS_BATCH_SIZE", "16"))
            method = os.environ.get("AKS_METHOD", "aks")
            max_iterations = int(os.environ.get("AKS_MAX_ITERATIONS", "3"))
            gap_threshold = float(os.environ.get("AKS_GAP_THRESHOLD", "0.2"))

            start_time = time.time()
            with torch.no_grad():
                with nvtx.annotate(message="Encode text", color="red"):
                    text_inputs = clip_processor(text=question, return_tensors="pt").to(
                        device)  # FIXME: synchronization

                visual_features = []
                candidate_visual_indices = []
                for batch_candidate_visual_indices in batched(aks_cand_frame_indices(vr, sampling_rate), batch_size):
                    # 1. Prepare candidate visuals
                    with nvtx.annotate(message="Prepare candidate visuals", color="green"):
                        frames = vr.get_batch(batch_candidate_visual_indices)
                    with nvtx.annotate(message="aks_clip_processor", color="red"):
                        candidate_visual = clip_processor(images=frames, return_tensors="pt")

                    # 2. Score candidate visuals
                    with nvtx.annotate(message="Score candidate visuals 1", color="green"):
                        features = clip_model.get_image_features(
                            pixel_values=candidate_visual["pixel_values"],
                        )
                        visual_features.append(features)
                        candidate_visual_indices.extend(batch_candidate_visual_indices)

                with nvtx.annotate(message="Score candidate visuals 2", color="yellow"):
                    visual_features = torch.cat(visual_features, dim=0)
                    text_features = clip_model.get_text_features(**text_inputs)
                    clip_scores = cosine_similarity(text_features, visual_features)

                    clip_scores = clip_scores.cpu().numpy()  # FIXME: synchronization

                # 3. Select frames
                with nvtx.annotate(message="Select visuals", color="green"):
                    selected_frame_indices = select_visuals(
                        clip_scores,
                        candidate_visual_indices,
                        nframes,
                        method=method,
                        max_iterations=max_iterations,
                        gap_threshold=gap_threshold,
                    )
                    selected_frame_indices.sort()

            idx = selected_frame_indices

            end_time = time.time()
            print(f"!!! AKS end | elapsed time: {end_time - start_time}s:")

        elif frame_selection_method == "byteclip":
            from byteclip.inference.segment_selection.byteclip.visual_utils import byteclip_preprocess
            from byteclip.inference.segment_selection.byteclip.selection_utils import select_segments, sample_frames_from_segments

            selection_method = os.environ.get("BYTECLIP_SELECTION_METHOD", "aks")
            num_segments = int(os.environ.get("BYTECLIP_NUM_SEGMENTS", "32"))
            max_iterations = int(os.environ.get("AKS_MAX_ITERATIONS", "3"))
            gap_threshold = float(os.environ.get("AKS_GAP_THRESHOLD", "0.2"))
            selection_params = dict(
                max_iterations=max_iterations,
                gap_threshold=gap_threshold,
            )

            print("!!! Byteclip start " + "=" * 80)
            start_time = time.time()

            key_indices = sorted(vr_cpu.get_key_indices())
            key_indices.append(len(vr_cpu))

            pad_token_id = byteclip_model.config.pad_token_id
            cls_token_id = byteclip_model.config.cls_token_id
            max_length = byteclip_model.config.max_length

            batch_size = int(os.environ.get("BYTECLIP_BATCH_SIZE", "16"))

            with torch.no_grad():
                with nvtx.annotate(message="Encode text", color="red"):
                    text_inputs = clip_processor(text=question, return_tensors="pt").to(device)  # FIXME: synchronization
                    text_features = clip_model.get_text_features(**text_inputs)

                visual_features = []
                candidate_visual_indices = []
                cand_gop_indices = np.arange(len(h264.keyframe_info)).tolist()
                for batch_candidate_visual_indices in batched(cand_gop_indices, batch_size):
                    # 1. Prepare candidate visuals
                    with nvtx.annotate(message="1. Preparing candidate visuals...", color="yellow"):
                        bytestreams = h264.get_batch(batch_candidate_visual_indices)
                        candidate_visual = byteclip_preprocess(bytestreams, cls_token_id, pad_token_id)
                        candidate_visual = {k: v.to(device) for k, v in candidate_visual.items()}

                    # 2-1. Score candidate visuals
                    with nvtx.annotate(message="2-1. Scoring candidate visuals (Visual Embedding)...", color="blue"):
                        features = byteclip_model(**candidate_visual).byte_embeds
                        visual_features.append(features)
                        candidate_visual_indices.extend(batch_candidate_visual_indices)

                # 2-2. Score candidate visuals
                with nvtx.annotate(message="2-2. Scoring candidate visuals (Cosine Similarity)...", color="green"):
                    visual_features = torch.cat(visual_features, dim=0)
                    clip_scores = cosine_similarity(text_features, visual_features)
                    clip_scores = clip_scores.cpu().numpy()

                # 3. Select segments
                with nvtx.annotate(message="3. Selecting segments...", color="yellow"):
                    selected_segment_indices, selected_segment_scores, branch_info = \
                        select_segments(
                            clip_scores,
                            candidate_visual_indices,  # NOTE: candidate "segment" indices
                            selection_method,
                            num_segments,
                            selection_params
                        )

                # 4. Sample frames from segments
                with nvtx.annotate(message="4. Sampling frames from segments...", color="red"):
                    selected_frame_indices = sample_frames_from_segments(
                        selected_segment_scores,
                        selected_segment_indices,
                        key_indices,
                        nframes
                    )

                print(f'{len(selected_frame_indices)} frames are '
                      f'sampled from {len(selected_segment_indices)} segments')
                print(selected_frame_indices)
                print('-' * 30)

            idx = selected_frame_indices

            end_time = time.time()
            print(f"!!! Byteclip end | elapsed time: {end_time - start_time}s:")

        start_time = time.time()
        with nvtx.annotate(message="Fetch frames", color="red"):
            video = vr.get_batch(idx)
            assert isinstance(video, torch.Tensor), f"video should be torch.Tensor, but got {type(video)}"
            video = video.permute(0, 3, 1, 2)  # Convert to TCHW format
            nframes, _, height, width = video.shape
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
            max_pixels = max(
                min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
                int(min_pixels * 1.05),
            )
            max_pixels_supposed = ele.get("max_pixels", max_pixels)
            if max_pixels_supposed > max_pixels:
                logger.warning(
                    f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
                )
            max_pixels = min(max_pixels_supposed, max_pixels)
            if "resized_height" in ele and "resized_width" in ele:
                resized_height, resized_width = smart_resize(
                    ele["resized_height"],
                    ele["resized_width"],
                    factor=image_factor,
                )
            else:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            video = torchvision.transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

        end_time = time.time()
        print(f"!!! Fetch frames end | elapsed time: {end_time - start_time}s:")

    return video


# Compatible with Qwen2VL and Qwen2_5VL
class Qwen2_5VLImageProcessor(SGLangBaseProcessor):
    models = [Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # The regex that matches expanded image tokens.
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
        ).build(_processor)

        self.frame_selection_method = os.environ.get("FRAME_SELECTION_METHOD", "uniform")
        print(f"!!! Using frame selection method: {self.frame_selection_method}")

        self.frame_selection_args = {}
        if self.frame_selection_method in ["aks", "byteclip"]:
            from transformers import CLIPProcessor, CLIPModel
            aks_model_card = os.environ.get("CLIP_MODEL_CARD")
            assert aks_model_card is not None, "CLIP_MODEL_CARD is not set"
            self.frame_selection_args = dict(
                clip_processor=CLIPProcessor.from_pretrained(aks_model_card, use_fast=True),
                clip_model=CLIPModel.from_pretrained(aks_model_card, device_map={
                    '': 0
                }).eval()
            )
            if self.frame_selection_method == "byteclip":
                from byteclip.models.modeling_modernbytebert import ModernByteBertModelWithProjection
                config_path = os.environ.get("BYTECLIP_CONFIG_PATH")
                self.frame_selection_args |= dict(
                    byteclip_model=ModernByteBertModelWithProjection.from_config(config_path).to('cuda:0'),
                )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Qwen-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [resize_image_async(image) for image in base_output.images]
            base_output.images = await asyncio.gather(*resize_tasks)

        videos = None
        if base_output.videos:
            base_output.videos = [
                await preprocess_video(
                    video,
                    frame_selection_method=self.frame_selection_method,
                    **self.frame_selection_args
                )
                for video in base_output.videos
            ]

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.hf_config.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=getattr(ret, "second_per_grid_ts", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
