# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch

from src.eunms import Model_Type, Scheduler_Type

@dataclass
class RunConfig:
    model_type: Model_Type = Model_Type.SDXL_Turbo
    scheduler_type: Scheduler_Type = Scheduler_Type.EULER

    seed: int = 7865
    num_inference_steps: int = 4
    num_inversion_steps: int = 4

    inv_guidance_scale: float = 1.5
    guidance_scale: float = 5.0
    style_guidance_scale: float = 0.0
    content_guidance_scale: float = 0.0

    inv_style_guidance_scale: float = 0.0
    inv_content_guidance_scale: float = 0.0
    inv_neg_style_guidance_scale: float = 0.0
    inv_neg_content_guidance_scale: float = 0.0

    get_grad_guidance: bool = True
    inv_guidance: float = 0.9

    num_renoise_steps: int = 9
    max_num_renoise_steps_first_step: int = 5
    inversion_max_step: float = 1.0

    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    useIP: bool = True

    resolution: int = 1024

    # Average Parameters
    average_latent_estimations: bool = True
    average_first_step_range: tuple[int, int] = (0, 5)
    average_step_range: tuple[int, int] = (8, 10)

    # Noise Regularization
    noise_regularization_lambda_ac: float = 20.0
    noise_regularization_lambda_kl: float = 0.065
    noise_regularization_num_reg_steps: int = 4
    noise_regularization_num_ac_rolls: int = 5

    # Noise Correction
    perform_noise_correction: bool = True

    # Model paths
    # These defaults use the official Hugging Face repos and are downloaded on demand.
    tile_controlnet_path: str = "xinsir/controlnet-tile-sdxl-1.0"
    canny_controlnet_path: str = "TheMistoAI/MistoLine"
    depth_controlnet_path: str = "diffusers/controlnet-depth-sdxl-1.0-small"

    canny_controlnet_path_sd15: str = "lllyasviel/control_v11p_sd15_canny"
    depth_controlnet_path_sd15: str = "lllyasviel/control_v11f1p_sd15_depth"

    IP_path: str = "h94/IP-Adapter"
    clip_model_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_path: str = "./CSD_Score/models/checkpoint.pth"

    base_model_path_sd15: str = "runwayml/stable-diffusion-v1-5"
    base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"

    control_type: str = "tile_canny"
    choose_pipeline: str = ""

    # Image parameters
    result_path: str = "./results"
    style_image_dir: str = "path/to/your/style/image/file"
    content_image_dir: str = "path/to/your/style/image/content/file"

    def __post_init__(self):
        pass
