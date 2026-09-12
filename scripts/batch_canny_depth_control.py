import argparse
from pathlib import Path
import random
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download, snapshot_download
import torch
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import AutoProcessor, Blip2ForConditionalGeneration, CLIPVisionModelWithProjection

import infer_style as style_impl
from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
from src.config import RunConfig
from src.eunms import Model_Type, Scheduler_Type


CONTENT_PROMPT = "use the composition from the image"
STYLE_PROMPT = "use the style from the image"

MODEL_TYPE_CHOICES = [member.name for member in Model_Type]
SCHEDULER_TYPE_CHOICES = [member.name for member in Scheduler_Type]
CONTROL_TYPE_CHOICES = ["tile", "canny", "depth", "combine", "tile_canny"]


def normalize_format(fmt: str) -> str:
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch StyleSSP transfer using the infer_style pipeline.")

    parser.add_argument("--content_dir", required=True, type=Path, help="Directory containing source images.")
    parser.add_argument("--format", required=True, help="Image extension to process, e.g. .jpeg, jpeg, .png.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory receiving the generated images.")
    parser.add_argument(
        "--style_image",
        type=Path,
        default=None,
        help="Reference style image. If omitted, each input image is reused as its own style reference.",
    )
    parser.add_argument(
        "--style_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of style images. If provided, one style image is selected at random for each "
            "content image. Matching seg_dict files are used when present, but plain images are also supported."
        ),
    )

    parser.add_argument("--model_type", type=str, default="SDXL", choices=MODEL_TYPE_CHOICES)
    parser.add_argument("--scheduler_type", type=str, default="DDIM", choices=SCHEDULER_TYPE_CHOICES)
    parser.add_argument("--choose_pipeline", type=str, default="", choices=["", "sd15"])
    parser.add_argument("--control_type", type=str, default="tile_canny", choices=CONTROL_TYPE_CHOICES)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_inversion_steps", type=int, default=50)
    parser.add_argument("--num_renoise_steps", type=int, default=1)
    parser.add_argument("--max_num_renoise_steps_first_step", type=int, default=5)
    parser.add_argument("--inv_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Direct prompt strength.")
    parser.add_argument("--style_guidance_scale", type=float, default=0.0, help="Direct style influence scale.")
    parser.add_argument(
        "--content_guidance_scale",
        type=float,
        default=0.0,
        help="Direct structure/content influence scale.",
    )
    parser.add_argument("--inv_style_guidance_scale", type=float, default=0.0)
    parser.add_argument("--inv_content_guidance_scale", type=float, default=0.0)
    parser.add_argument("--inv_neg_style_guidance_scale", type=float, default=0.0)
    parser.add_argument("--inv_neg_content_guidance_scale", type=float, default=0.0)
    parser.add_argument("--inv_guidance", type=float, default=0.9)
    parser.add_argument(
        "--content_image_prompt",
        type=str,
        default=None,
        help=(
            "Optional prompt override for content images. If omitted, a prompt is inferred per input image "
            "using BLIP."
        ),
    )

    args = parser.parse_args()

    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be > 0.")
    if args.num_inversion_steps <= 0:
        parser.error("--num_inversion_steps must be > 0.")
    if args.num_renoise_steps <= 0:
        parser.error("--num_renoise_steps must be > 0.")
    if args.max_num_renoise_steps_first_step <= 0:
        parser.error("--max_num_renoise_steps_first_step must be > 0.")
    if args.resolution <= 0:
        parser.error("--resolution must be > 0.")
    if args.style_image is not None and not args.style_image.is_file():
        parser.error(f"Style image does not exist: {args.style_image}")
    if args.style_dir is not None and not args.style_dir.is_dir():
        parser.error(f"Style directory does not exist: {args.style_dir}")

    return args


def collect_content_images(content_dir: Path, image_format: str) -> List[Path]:
    input_files = sorted(
        path for path in content_dir.iterdir() if path.is_file() and path.suffix.lower() == image_format
    )
    if not input_files:
        raise SystemExit(f"No files with format '{image_format}' found in {content_dir}")
    return input_files


def resolve_struct_seg_dict(content_img: Path) -> Optional[Path]:
    candidates = [
        content_img.with_suffix(".pth"),
        content_img.parent / "seg_dict" / f"{content_img.stem}.pth",
    ]
    if content_img.parent.name.lower() == "images":
        candidates.append(content_img.parent.parent / "seg_dict" / f"{content_img.stem}.pth")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def resolve_style_seg_dict(style_img: Path) -> Optional[Path]:
    candidates = [
        style_img.parent / "seg_dict.pth",
        style_img.with_suffix(".pth"),
        style_img.parent / "seg_dict" / f"{style_img.stem}.pth",
    ]
    if style_img.parent.name.lower() == "images":
        candidates.append(style_img.parent.parent / "seg_dict" / f"{style_img.stem}.pth")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def collect_style_images(style_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    style_refs: List[Tuple[Path, Optional[Path]]] = []

    for path in sorted(style_dir.iterdir()):
        if path.is_dir():
            image_candidates = sorted(path.glob("image.*"))
            seg_candidate = path / "seg_dict.pth"
            if image_candidates:
                style_refs.append((image_candidates[0], seg_candidate if seg_candidate.is_file() else None))
            continue

        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            style_refs.append((path, resolve_style_seg_dict(path)))

    if not style_refs:
        raise SystemExit(
            "No valid style references found in style_dir. Expected image files, optionally with seg_dict.pth next to them, "
            "or subfolders containing image.* with an optional seg_dict.pth."
        )

    return style_refs


def choose_base_model_path(args: argparse.Namespace) -> str:
    return "runwayml/stable-diffusion-v1-5" if args.choose_pipeline == "sd15" else "stabilityai/stable-diffusion-xl-base-1.0"


def ensure_hf_checkpoints(args: argparse.Namespace) -> None:
    print("Prefetching Hugging Face checkpoints for StyleSSP...")
    checkpoints = [
        "Salesforce/blip2-flan-t5-xl",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "Intel/dpt-hybrid-midas",
        "madebyollin/sdxl-vae-fp16-fix",
        "h94/IP-Adapter",
        "TheMistoAI/MistoLine",
        "xinsir/controlnet-tile-sdxl-1.0",
        "diffusers/controlnet-depth-sdxl-1.0-small",
        choose_base_model_path(args),
    ]
    for repo_id in checkpoints:
        print(f"  - {repo_id}")
        snapshot_download(repo_id=repo_id)

    models_dir = PROJECT_ROOT / "checkpoints" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["ip-adapter-instruct-sdxl.bin", "ip-adapter-instruct-sd15.bin"]:
        print(f"  - CiaraRowles/IP-Adapter-Instruct::{filename}")
        hf_hub_download(
            repo_id="CiaraRowles/IP-Adapter-Instruct",
            filename=filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )

    print("All Hugging Face checkpoints are available in local cache.")


def load_captioning_models():
    model_id = "Salesforce/blip2-flan-t5-xl"
    caption_processor = AutoProcessor.from_pretrained(model_id)
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        load_in_8bit=False,
        torch_dtype=torch.float16,
    )
    caption_model.eval()
    return caption_processor, caption_model


def load_depth_models():
    depth_estimator = style_impl.DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = style_impl.DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    return depth_estimator, feature_extractor


def build_controlnet_bundle(config: RunConfig, device: str):
    if config.control_type == "tile":
        controlnet = ControlNetModel.from_pretrained(
            config.tile_controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
        return controlnet, 0.25

    if config.control_type == "canny":
        controlnet = ControlNetModel.from_pretrained(
            config.canny_controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        return controlnet, 0.2

    if config.control_type == "depth":
        controlnet = ControlNetModel.from_pretrained(
            config.depth_controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        return controlnet, 0.4

    if config.control_type == "combine":
        controlnet = [
            ControlNetModel.from_pretrained(
                config.depth_controlnet_path,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device),
            ControlNetModel.from_pretrained(
                config.canny_controlnet_path,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device),
        ]
        return controlnet, [0.4, 0.4]

    controlnet = [
        ControlNetModel.from_pretrained(
            config.tile_controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device),
        ControlNetModel.from_pretrained(
            config.canny_controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device),
    ]
    return controlnet, [0.25, 0.40]


def build_control_image(
    config: RunConfig,
    content_image_path: Path,
    content_image: Image.Image,
    depth_models: Optional[Tuple[object, object]],
):
    if config.control_type == "tile":
        return load_image(str(content_image_path)).resize((config.resolution, config.resolution))

    if config.control_type == "canny":
        input_image_cv2 = cv2.imread(str(content_image_path))
        input_image_cv2 = np.array(input_image_cv2)
        anyline_image = style_impl.get_canny_map(input_image_cv2)
        return anyline_image.resize((config.resolution, config.resolution))

    if config.control_type == "depth":
        if depth_models is None:
            raise ValueError("Depth models are required for control_type=depth")
        depth_image = style_impl.get_depth_map(
            content_image,
            depth_estimator=depth_models[0],
            feature_extractor=depth_models[1],
            target_resolution=config.resolution,
        )
        return depth_image.resize((config.resolution, config.resolution))

    if config.control_type == "combine":
        if depth_models is None:
            raise ValueError("Depth models are required for control_type=combine")
        depth_image = style_impl.get_depth_map(
            content_image,
            depth_estimator=depth_models[0],
            feature_extractor=depth_models[1],
            target_resolution=config.resolution,
        )
        cond_depth_image = depth_image.resize((config.resolution, config.resolution))

        input_image_cv2 = cv2.imread(str(content_image_path))
        input_image_cv2 = np.array(input_image_cv2)
        anyline_image = style_impl.get_canny_map(input_image_cv2)
        cond_canny_image = anyline_image.resize((config.resolution, config.resolution))
        return [cond_depth_image, cond_canny_image]

    cond_image = load_image(str(content_image_path))
    cond_tile_image = cond_image.resize((config.resolution, config.resolution))

    input_image_cv2 = cv2.imread(str(content_image_path))
    input_image_cv2 = np.array(input_image_cv2)
    anyline_image = style_impl.get_canny_map(input_image_cv2)
    cond_canny_image = anyline_image.resize((config.resolution, config.resolution))
    return [cond_tile_image, cond_canny_image]


def make_run_config(args: argparse.Namespace, content_img: Path, style_img: Path, output_dir: Path) -> RunConfig:
    #import inspect
    #print("RunConfig loaded from:", inspect.getfile(RunConfig))
    #print("RunConfig fields:", list(RunConfig.__dataclass_fields__.keys()))

    cfg = RunConfig(
        model_type=Model_Type[args.model_type],
        scheduler_type=Scheduler_Type[args.scheduler_type],
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        num_inversion_steps=args.num_inversion_steps,
        inv_guidance_scale=args.inv_guidance_scale,
        guidance_scale=args.guidance_scale,
        style_guidance_scale=args.style_guidance_scale,
        content_guidance_scale=args.content_guidance_scale,
        inv_style_guidance_scale=args.inv_style_guidance_scale,
        inv_content_guidance_scale=args.inv_content_guidance_scale,
        inv_neg_style_guidance_scale=args.inv_neg_style_guidance_scale,
        inv_neg_content_guidance_scale=args.inv_neg_content_guidance_scale,
        inv_guidance=args.inv_guidance,
        num_renoise_steps=args.num_renoise_steps,
        max_num_renoise_steps_first_step=args.max_num_renoise_steps_first_step,
        resolution=args.resolution,
        control_type=args.control_type,
        choose_pipeline=args.choose_pipeline,
        style_image_dir=str(style_img),
        content_image_dir=str(content_img),
        result_path=str(output_dir),
    )
    cfg.base_model_path = choose_base_model_path(args)
    cfg.__dict__["content_image_prompt"] = args.content_image_prompt
    return cfg


def build_bootstrap_style(args: argparse.Namespace, input_files: List[Path], style_refs: Optional[List[Tuple[Path, Optional[Path]]]]) -> Tuple[Path, Optional[Path]]:
    if style_refs:
        return style_refs[0]
    if args.style_image is not None:
        style_seg = resolve_style_seg_dict(args.style_image)
        return args.style_image, style_seg
    style_img = input_files[0]
    return style_img, resolve_style_seg_dict(style_img)


def main() -> None:
    args = parse_args()

    content_dir = args.content_dir
    output_dir = args.output_dir
    image_format = normalize_format(args.format)

    if not content_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {content_dir}")

    ensure_hf_checkpoints(args)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = collect_content_images(content_dir, image_format)
    style_refs = collect_style_images(args.style_dir) if args.style_dir is not None else None

    if style_refs is not None:
        print(f"Using random style reference per image from: {args.style_dir}")
        print(f"Discovered {len(style_refs)} style reference(s); segmentation dictionaries will be used when available.")
    elif args.style_image is None:
        print("No --style_image provided: using each input image as its own style reference.")
    else:
        print(f"Using global style reference: {args.style_image}")

    print(f"Found {len(input_files)} input file(s).")
    print(f"Output folder: {output_dir}")
    print(
        f"Influence controls: guidance_scale={args.guidance_scale}, style_guidance_scale={args.style_guidance_scale}, "
        f"content_guidance_scale={args.content_guidance_scale}, inv_guidance_scale={args.inv_guidance_scale}"
    )
    if args.content_image_prompt is None:
        print("Content prompt mode: infer per-image using BLIP captions.")
    else:
        print(f"Content prompt mode: using CLI override for all images: {args.content_image_prompt}")

    bootstrap_style_img, bootstrap_style_seg = build_bootstrap_style(args, input_files, style_refs)
    bootstrap_content_img = input_files[0]
    bootstrap_content_seg = resolve_struct_seg_dict(bootstrap_content_img)

    bootstrap_cfg = make_run_config(args, bootstrap_content_img, bootstrap_style_img, output_dir)
    if bootstrap_style_seg is not None:
        bootstrap_cfg.app_seg_dict = str(bootstrap_style_seg)
    if bootstrap_content_seg is not None:
        bootstrap_cfg.struct_seg_dict = str(bootstrap_content_seg)

    caption_processor, caption_model = load_captioning_models()
    depth_models = None
    if args.control_type in {"depth", "combine"}:
        depth_models = load_depth_models()

    ip_instruct_model = style_impl.init_models(bootstrap_cfg)
    pipe_inversion, pipe_inference_ref = style_impl.get_pipes(
        bootstrap_cfg.model_type,
        bootstrap_cfg.scheduler_type,
        device=bootstrap_cfg.device,
        model_name=bootstrap_cfg.base_model_path,
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=bootstrap_cfg.dtype,
    ).to(bootstrap_cfg.device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=bootstrap_cfg.dtype).to(bootstrap_cfg.device)
    controlnet, controlnet_conditioning_scale = build_controlnet_bundle(bootstrap_cfg, bootstrap_cfg.device)

    pipe_inference = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        bootstrap_cfg.base_model_path,
        controlnet=controlnet,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(bootstrap_cfg.device)
    pipe_inference.scheduler = UniPCMultistepScheduler.from_config(pipe_inference.scheduler.config)
    pipe_inference.unet.enable_gradient_checkpointing()
    pipe_inference.load_ip_adapter(
        bootstrap_cfg.IP_path,
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.safetensors",
        image_encoder_folder=None,
    )
    pipe_inference.set_ip_adapter_scale({"up": {"block_0": [0.0, 2.5, 0.0]}})

    rng = random.Random(args.seed)

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        struct_seg = resolve_struct_seg_dict(input_path)

        if style_refs is not None:
            style_path, style_seg = rng.choice(style_refs)
        else:
            style_path = args.style_image if args.style_image is not None else input_path
            style_seg = resolve_style_seg_dict(style_path)

        print(f"[{index}/{len(input_files)}] {input_path.name} style={style_path.name} -> {output_path}")

        cfg = make_run_config(args, input_path, style_path, output_dir)
        if struct_seg is not None:
            cfg.struct_seg_dict = str(struct_seg)
        if style_seg is not None:
            cfg.app_seg_dict = str(style_seg)

        style_image = Image.open(style_path).convert("RGB").resize((cfg.resolution, cfg.resolution))
        content_image = Image.open(input_path).convert("RGB")
        content_image = content_image.resize((cfg.resolution, cfg.resolution))

        style_embeddings_instruct = ip_instruct_model.get_decouple_embeds(
            pil_image=style_image,
            prompt="",
            query=STYLE_PROMPT,
        )
        style_content_embeddings = ip_instruct_model.get_decouple_embeds(
            pil_image=style_image,
            prompt="",
            query=CONTENT_PROMPT,
        )

        if cfg.content_image_prompt is None:
            content_image_prompt = style_impl.generate_caption(
                content_image,
                caption_processor=caption_processor,
                caption_model=caption_model,
            )
        else:
            content_image_prompt = cfg.content_image_prompt
        content_embeddings_instruct = ip_instruct_model.get_decouple_embeds(
            pil_image=content_image,
            prompt="",
            query=CONTENT_PROMPT,
        )
        content_style_instruct = ip_instruct_model.get_decouple_embeds(
            pil_image=content_image,
            prompt="",
            query=STYLE_PROMPT,
        )
        print(content_image_prompt)

        entire_mask = Image.new("RGB", (cfg.resolution, cfg.resolution), color=(255, 255, 255))

        _, inv_latent, _, all_latents = style_impl.invert(
            content_image,
            content_image_prompt,
            cfg,
            pipe_inversion=pipe_inversion,
            pipe_inference=pipe_inference_ref,
            do_reconstruction=False,
            feature_extractor=ip_instruct_model,
            style_embedding=style_embeddings_instruct,
            content_embedding=content_embeddings_instruct,
            neg_style_embedding=content_style_instruct,
            neg_content_embedding=style_content_embeddings,
            enable_guidance=False,
            used_NPI_guidance=True,
        )

        _, latent_l, _ = style_impl.freq_exp(inv_latent, d_s=0.3, d_t=0.9, alpha=0.7, filter_type="gaussian_b")
        latent_l = latent_l.to(inv_latent.dtype)

        del all_latents
        torch.cuda.empty_cache()

        control_image = build_control_image(cfg, input_path, content_image, depth_models)

        output = pipe_inference(
            prompt=content_image_prompt,
            negative_prompt="watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            num_inference_steps=cfg.num_inference_steps,
            eta=1.0,
            mask_image=entire_mask,
            image=content_image,
            control_image=control_image,
            ip_adapter_image=style_image,
            generator=torch.Generator(device="cpu").manual_seed(cfg.seed),
            latents=latent_l,
            guidance_scale=cfg.guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            npi_interp=0.5,
            style_embeddings_instruct=style_embeddings_instruct,
            content_embeddings_instruct=content_embeddings_instruct,
            style_guidance_scale=cfg.style_guidance_scale,
            content_guidance_scale=cfg.content_guidance_scale,
            ip_instruct_model=ip_instruct_model,
            CSD_model=None,
            inv_guidance=cfg.inv_guidance,
            feature_extractor=ip_instruct_model,
            do_NPI=False,
        ).images[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
