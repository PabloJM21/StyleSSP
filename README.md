# [CVPR 2025] StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer

### [Arxiv](https://arxiv.org/abs/2501.11319)

![imgs](assets/ours.jpg)

## Usage

**To run the code, please follow these step:**

1. [Download](#download)
2. [Setup](#environment-setup)
3. [Run](#run)

### Download

This project contains contributions from [ControlNet](https://github.com/lllyasviel/ControlNet) and [IP-Adapter-Instruct](https://github.com/unity-research/IP-Adapter-Instruct), licensed under the Apache License 2.0. StyleSSP now prefetches the required Hugging Face checkpoints automatically when you run the batch CLI, so you do not need to download them manually first.

The runtime loads these model families:

* `Salesforce/blip2-flan-t5-xl` for style captioning.
* `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` for IP-Adapter-Instruct vision features.
* `Intel/dpt-hybrid-midas` for depth conditioning.
* `h94/IP-Adapter` plus `CiaraRowles/IP-Adapter-Instruct` for style-aware instruction embeddings.
* `TheMistoAI/MistoLine`, `xinsir/controlnet-tile-sdxl-1.0`, and `diffusers/controlnet-depth-sdxl-1.0-small` for the structure-conditioning path.
* `madebyollin/sdxl-vae-fp16-fix` for the SDXL VAE.
* `stabilityai/stable-diffusion-xl-base-1.0` by default, or `runwayml/stable-diffusion-v1-5` when `--choose_pipeline sd15` is selected.

The first CLI run stores the checkpoints under `checkpoints/` and the Hugging Face cache.

### Environment Setup

```
conda env create -f environment.yaml
conda activate StyleSSP
pip install git+https://github.com/openai/CLIP.git
```

The environment files are aligned with Python 3.11.6, and `xformers` is no longer required.

### Run

The main batch entrypoint is:

```
python scripts/batch_canny_depth_control.py ...
```

Required flags:

* `--content_dir`
* `--format`
* `--output_dir`

Style reference flags:

* Use `--style_image` for one global style reference.
* Use `--style_dir` for random per-image style selection from a style pool. When `--style_dir` is set, it takes priority over `--style_image`.

Prompt and influence flags:

* `--content_image_prompt`: if set, this prompt is used for all images; if omitted, BLIP infers one prompt per content image.
* `--guidance_scale`: prompt strength in generation.
* `--style_guidance_scale`: style influence.
* `--content_guidance_scale`: structure/content influence.
* `--inv_guidance_scale`, `--inv_guidance`, `--inv_style_guidance_scale`, `--inv_content_guidance_scale`, `--inv_neg_style_guidance_scale`, `--inv_neg_content_guidance_scale`: inversion guidance controls.

Model and runtime flags:

* `--model_type` (`SDXL` by default)
* `--scheduler_type` (`DDIM` by default)
* `--choose_pipeline` (`""` default or `sd15`)
* `--control_type` (`tile`, `canny`, `depth`, `combine`, `tile_canny`)
* `--resolution`, `--seed`, `--num_inference_steps`, `--num_inversion_steps`, `--num_renoise_steps`, `--max_num_renoise_steps_first_step`

Complete example with all relevant flags:

```
python scripts/batch_canny_depth_control.py \
  --content_dir data/content \
  --format .png \
  --output_dir results \
  --style_image data/style/7.jpg \
  --model_type SDXL \
  --scheduler_type DDIM \
  --choose_pipeline "" \
  --control_type tile_canny \
  --resolution 1024 \
  --seed 1234 \
  --num_inference_steps 50 \
  --num_inversion_steps 50 \
  --num_renoise_steps 1 \
  --max_num_renoise_steps_first_step 5 \
  --inv_guidance_scale 1.5 \
  --inv_guidance 0.9 \
  --inv_style_guidance_scale 0.0 \
  --inv_content_guidance_scale 0.0 \
  --inv_neg_style_guidance_scale 0.0 \
  --inv_neg_content_guidance_scale 0.0 \
  --content_image_prompt "a realistic sunny landscape with natural colors" \
  --guidance_scale 5.0 \
  --style_guidance_scale 0.35 \
  --content_guidance_scale 0.35
```

The original single-image script remains available via `python infer_style.py`, but the batch CLI above is the recommended workflow.

### Pipeline Workflow

The batch CLI runs in five stages:

1. Checkpoint prefetch: downloads and caches all required models (`Salesforce/blip2-flan-t5-xl`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `Intel/dpt-hybrid-midas`, `madebyollin/sdxl-vae-fp16-fix`, `h94/IP-Adapter`, `CiaraRowles/IP-Adapter-Instruct`, `TheMistoAI/MistoLine`, `xinsir/controlnet-tile-sdxl-1.0`, `diffusers/controlnet-depth-sdxl-1.0-small`, and the selected base model).
2. Style/content feature setup: loads the style reference image and computes decoupled style/content embeddings with IP-Adapter-Instruct.
3. Content prompt setup: uses `--content_image_prompt` if provided; otherwise infers one prompt per content image using BLIP.
4. Inversion and latent editing: inverts the content image into latent space and applies the StyleSSP frequency-domain latent manipulation.
5. Controlled reconstruction: reconstructs with ControlNet (`tile`, `canny`, `depth`, or combinations) and applies the direct influence scales (`guidance_scale`, `style_guidance_scale`, `content_guidance_scale`).

The optional seg dicts affect only the style-reference branch:

* If `--style_dir` contains only images, the CLI uses those images directly as style references and samples one at random for each content image. The batch run still works, and the style transfer is driven by the style image plus IP-Adapter-Instruct embeddings.
* If matching `seg_dict.pth` files are present for the style images, the CLI uses them to provide a more structured style-conditioning signal. That lets the style branch keep region-level semantics more consistently when the reference image has multiple parts, while still relying on the same BLIP and IP-Adapter checkpoints.
* If no seg dict is available for a style image, the code falls back to the image-only path automatically. In other words, seg dicts are an enhancement, not a requirement.

The content-side segmentation dictionary remains required, because it tells StyleSSP how to preserve the structure of the input image during inversion and reconstruction. In practice, that structure is then combined with the depth or edge ControlNet checkpoint selected by `--control_type`.

### Priority Examples

Structure priority:

```
python scripts/batch_canny_depth_control.py \
  --content_dir data/content \
  --format .png \
  --output_dir results_structure \
  --style_image data/style/7.jpg \
  --model_type SDXL \
  --scheduler_type DDIM \
  --choose_pipeline "" \
  --resolution 1024 \
  --seed 1234 \
  --num_inference_steps 50 \
  --num_inversion_steps 50 \
  --num_renoise_steps 1 \
  --max_num_renoise_steps_first_step 5 \
  --inv_guidance_scale 1.5 \
  --inv_guidance 0.9 \
  --inv_style_guidance_scale 0.0 \
  --inv_content_guidance_scale 0.0 \
  --inv_neg_style_guidance_scale 0.0 \
  --inv_neg_content_guidance_scale 0.0 \
  --content_image_prompt "a realistic sunny mountain landscape, high detail" \
  --guidance_scale 4.5 \
  --style_guidance_scale 0.1 \
  --content_guidance_scale 0.85 \
  --control_type depth
```

Balanced transfer:

```
python scripts/batch_canny_depth_control.py \
  --content_dir data/content \
  --format .png \
  --output_dir results_balanced \
  --style_image data/style/7.jpg \
  --model_type SDXL \
  --scheduler_type DDIM \
  --choose_pipeline "" \
  --resolution 1024 \
  --seed 1234 \
  --num_inference_steps 50 \
  --num_inversion_steps 50 \
  --num_renoise_steps 1 \
  --max_num_renoise_steps_first_step 5 \
  --inv_guidance_scale 1.5 \
  --inv_guidance 0.9 \
  --inv_style_guidance_scale 0.0 \
  --inv_content_guidance_scale 0.0 \
  --inv_neg_style_guidance_scale 0.0 \
  --inv_neg_content_guidance_scale 0.0 \
  --content_image_prompt "a realistic sunny landscape with natural colors" \
  --guidance_scale 5.0 \
  --style_guidance_scale 0.35 \
  --content_guidance_scale 0.35 \
  --control_type tile_canny
```

Style priority:

```
python scripts/batch_canny_depth_control.py \
  --content_dir data/content \
  --format .png \
  --output_dir results_style \
  --style_dir data/style \
  --model_type SDXL \
  --scheduler_type DDIM \
  --choose_pipeline "" \
  --resolution 1024 \
  --seed 1234 \
  --num_inference_steps 50 \
  --num_inversion_steps 50 \
  --num_renoise_steps 1 \
  --max_num_renoise_steps_first_step 5 \
  --inv_guidance_scale 1.5 \
  --inv_guidance 0.9 \
  --inv_style_guidance_scale 0.0 \
  --inv_content_guidance_scale 0.0 \
  --inv_neg_style_guidance_scale 0.0 \
  --inv_neg_content_guidance_scale 0.0 \
  --content_image_prompt "a realistic snowy landscape at golden hour" \
  --guidance_scale 4.0 \
  --style_guidance_scale 0.9 \
  --content_guidance_scale 0.15 \
  --control_type tile
```

Prompt priority:

```
python scripts/batch_canny_depth_control.py \
  --content_dir data/content \
  --format .png \
  --output_dir results_prompt \
  --style_image data/style/7.jpg \
  --model_type SDXL \
  --scheduler_type DDIM \
  --choose_pipeline "" \
  --resolution 1024 \
  --seed 1234 \
  --num_inference_steps 50 \
  --num_inversion_steps 50 \
  --num_renoise_steps 1 \
  --max_num_renoise_steps_first_step 5 \
  --inv_guidance_scale 1.5 \
  --inv_guidance 0.9 \
  --inv_style_guidance_scale 0.0 \
  --inv_content_guidance_scale 0.0 \
  --inv_neg_style_guidance_scale 0.0 \
  --inv_neg_content_guidance_scale 0.0 \
  --content_image_prompt "a realistic sunny landscape after rain, cinematic lighting" \
  --guidance_scale 7.5 \
  --style_guidance_scale 0.15 \
  --content_guidance_scale 0.2 \
  --control_type canny
```

Image-only style directory example (no style seg dict required):

```
python scripts/batch_canny_depth_control.py \
  --content_dir data/content \
  --format .png \
  --output_dir results_style_pool \
  --style_dir data/style \
  --model_type SDXL \
  --scheduler_type DDIM \
  --choose_pipeline "" \
  --control_type tile_canny \
  --resolution 1024 \
  --seed 1234 \
  --num_inference_steps 50 \
  --num_inversion_steps 50 \
  --num_renoise_steps 1 \
  --max_num_renoise_steps_first_step 5 \
  --inv_guidance_scale 1.5 \
  --inv_guidance 0.9 \
  --inv_style_guidance_scale 0.0 \
  --inv_content_guidance_scale 0.0 \
  --inv_neg_style_guidance_scale 0.0 \
  --inv_neg_content_guidance_scale 0.0 \
  --content_image_prompt "a realistic sunny valley landscape" \
  --guidance_scale 5.0 \
  --style_guidance_scale 0.5 \
  --content_guidance_scale 0.35
```

Then, run:
```
cd evaluation;
python eval_artfid.py --sty ../data_evl/style --cnt ../data_evl/content --tar ../data_evl/tar
```

## Citation
If you find our work useful, please consider citing and star:
```
@article{xu2025stylessp,
  title={StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer},
  author={Xu, Ruojun and Xi, Weijie and Wang, Xiaodi and Mao, Yongbo and Cheng, Zach},
  journal={arXiv preprint arXiv:2501.11319},
  year={2025}
}
```
