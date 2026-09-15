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


## StyleSSP Pipeline (Paper Version)

*(What the paper actually describes)*

StyleSSP consists of three major components, each mathematically defined in the paper:

### 1. DDIM Inversion + Frequency Manipulation (Section 4.1)

The paper begins by mapping the content image into the diffusion latent space using DDIM inversion:

$$z_T^c = \text{DDIM-Inv}(z_0^c)$$

Then it applies low‑frequency filtering to the inverted latent:

$$z_{T,L,\alpha}^c = \text{LPF}(z_T^c) + \mathcal{N}(0, \sigma^2) \cdot (1 - \alpha)$$

**Where:**
* **LPF** = low‑pass filter (frequency-domain smoothing)
* **$\alpha$** = blending factor between original and low‑frequency latent
* **$\sigma$** = noise strength for restoring stochasticity

This produces a latent that preserves layout but removes high‑frequency content, making style injection easier.

### 2. Negative Guidance via Inversion (Section 4.2)

StyleSSP introduces negative guidance inside the inversion process:

$$\hat{\epsilon}_\theta(z_t, t, C^+, C^-) = \epsilon_\theta(z_t, t, C^-) + \omega_i (\epsilon_\theta(z_t, t, C^+) - \epsilon_\theta(z_t, t, C^-))$$

**Where:**
* **$C^+$** = positive prompt embedding
* **$C^-$** = negative prompt embedding
* **$\omega_i$** = negative guidance scale

This pushes the latent away from undesired content and toward desired content during inversion, not just during denoising.

### 3. IP‑Instruct Negative Guidance (Section 4.2, Eq. 11)

The paper replaces text prompts with IP‑Instruct embeddings:

$$E^- = \text{concat}(\Phi(I_c)_s, \Phi(I_s)_c)$$

**Where:**
* **$\Phi(I_c)_s$** = style embedding of content image
* **$\Phi(I_s)_c$** = content embedding of style image

Then negative guidance becomes:

$$\hat{\epsilon}_\theta(z_t, t, C^+, E^-) = \epsilon_\theta(z_t, t, E^-) + \omega_i (\epsilon_\theta(z_t, t, C^+) - \epsilon_\theta(z_t, t, E^-))$$

This is the core innovation: use IP‑Instruct embeddings as negative guidance to push the latent toward the desired style while preserving content.

### 4. Style Injection (Section 4.3)

StyleSSP injects style features into specific UNet blocks, similar to InstantStyle:
* Inject style features into block 0 / block 1 / block 2
* Use IP‑Adapter or custom style injection
* Combine with ControlNet to preserve layout

This creates a controlled style transfer that respects content structure.

---

## 📘 Current SDXL Pipeline (Your Implementation)

*(What your code actually does today)*

Your batch script + `StableDiffusionXLControlNetInpaintPipeline` implements a subset of StyleSSP:

### 1. DDIM Inversion (now correct)
You now use:
* `MyDDIMScheduler` for inversion
* `UniPCMultistepScheduler` for inference

This is correct and stable.

### 2. Frequency Manipulation (partial)
You call:
```python
_, latent_l, _ = style_impl.freq_exp(inv_latent, ...)
```
**But:**
* No Gaussian noise term is added
* No $\alpha$‑blending is applied
* The filtered latent is not used inside inversion
* The filtered latent is not used inside inference
* The filtered latent is not used inside guidance

So this is not yet the paper’s method.

### 3. Negative Guidance (disabled)
Your inversion step currently:
* Does not compute $C^+$ / $C^-$
* Does not compute $E^-$
* Does not apply $\omega_i$
* Does not apply negative guidance inside DDIM inversion
* Has guidance block commented out

So Section 4.2 is not implemented.

### 4. IP‑Instruct Embeddings (computed but unused)
You compute:
* $\Phi(I_c)_s$
* $\Phi(I_s)_c$
* $\Phi(I_c)_c$
* $\Phi(I_s)_s$

**But:**
* You never concatenate them into $E^-$
* You never use $E^-$ inside UNet
* You never use $E^-$ inside inversion
* You never use $E^-$ inside inference

So Section 4.2 (Eq. 11) is not implemented.

### 5. Style Injection (not implemented)
You load IP‑Adapter:
```python
pipe_inference.load_ip_adapter(...)
```
**But:**
* You do not inject style into specific UNet blocks
* You do not modulate block injection scale
* You do not use InstantStyle‑style injection
* You do not use style injection during inversion

So Section 4.3 is not implemented.

### 6. ControlNet (correct)
Your pipeline uses:
* Tile / Canny / Depth ControlNet
* Correct conditioning scale
* Correct layout preservation

This matches the paper’s intent.

---

## 📘 How to Reproduce the Paper’s Behavior (Expansion Guidelines)

*(Exact steps you must implement)*

Below is the complete checklist to upgrade your SDXL pipeline to match StyleSSP.

### A. Implement Full Frequency Manipulation (Section 4.1)
Add:
* Low‑pass filter (already implemented)
* Gaussian noise term:
  $$z_{T,L,\alpha}^c = \text{LPF}(z_T^c) + \mathcal{N}(0, \sigma^2)(1 - \alpha)$$
* $\alpha$‑blending:
  $$z_T' = \alpha z_T^c + (1 - \alpha)z_{T,L,\alpha}^c$$

Use $z_T'$ as:
* inversion latent
* initial inference latent
* guidance latent

### B. Implement Negative Guidance via Inversion (Section 4.2)
Inside DDIM inversion:
* **Compute:**
  ```text
  C+ = positive prompt embedding
  C- = negative prompt embedding
  ```
* **Compute noise predictions:**
  ```text
  eps_pos = UNet(z_t, t, C+)
  eps_neg = UNet(z_t, t, C-)
  ```
* **Apply negative guidance:**
  $$\hat{\epsilon} = \epsilon_{\text{neg}} + \omega_i (\epsilon_{\text{pos}} - \epsilon_{\text{neg}})$$
* **Use `hat_eps` in DDIM step:**
  ```python
  step_out = scheduler.step(hat_eps, t, latents)
  ```

### C. Implement IP‑Instruct Negative Guidance (Eq. 11)
* **Compute:**
  ```text
  E- = concat(Phi(Ic)_s, Phi(Is)_c)
  ```
* **Compute noise predictions:**
  ```text
  eps_pos = UNet(z_t, t, C+)
  eps_neg = UNet(z_t, t, E-)
  ```
* **Apply:**
  $$\hat{\epsilon} = \epsilon_{E^-} + \omega_i (\epsilon_{C^+} - \epsilon_{E^-})$$
* Use `hat_eps` in DDIM step.

### D. Implement Style Injection (Section 4.3)
* **Choose block(s):** block 0 / block 1 / block 2
* **Inject style features:** from IP‑Adapter or from $\Phi(I_s)_s$
* **Use:**
  ```python
  down_block_additional_residuals
  mid_block_additional_residuals
  ```

### E. Integrate Frequency Manipulation + Negative Guidance + Style Injection
The full pipeline becomes:
1. DDIM inversion $\rightarrow z_T$
2. Frequency manipulation $\rightarrow z_T'$
3. Negative guidance inside inversion $\rightarrow \hat{\epsilon}$
4. Style injection inside UNet blocks
5. ControlNet layout preservation
6. UniPC inference with style/content guidance

### F. Update Main Pipeline
You must update:
* `inversion_step`
* `cond_fn`
* `main()` latent handling
* IP‑Adapter scale
* UNet block injection
* DDIM inversion math
* Negative guidance math
* Frequency manipulation usage

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
* `--scheduler_type` (`EULER` by default)
* `--choose_pipeline` (`""` default or `sd15`)
* `--control_type` (`tile`, `canny`, `depth`, `combine`, `tile_canny`)
* `--resolution`, `--seed`, `--num_inference_steps`, `--num_inversion_steps`, `--num_renoise_steps`, `--max_num_renoise_steps_first_step`

## Why inversion uses DDIM while inference uses UniPC

StyleSSP performs a *renoise inversion* step before denoising. That inversion process is not a forward diffusion sampler; it is a latent reconstruction step.

For SDXL inversion, the scheduler must be stable under reverse-time latent reconstruction. Euler Ancestral is a forward diffusion sampler: its update rule adds noise at each step and is designed for generation, not inversion. In practice, using `MyEulerAncestralDiscreteScheduler` for inversion causes the latent trajectory to explode and destabilize the renoise procedure.

The correct split is therefore:

- Inversion: `MyDDIMScheduler` for stable latent reconstruction and renoise compatibility.
- Inference: `UniPCMultistepScheduler` for denoising, which is already the correct sampler for the final generation pass.

This matches the project design:

- `pipe_inversion.scheduler = MyDDIMScheduler.from_config(...)`
- `pipe_inference.scheduler = UniPCMultistepScheduler.from_config(...)`

This separation is important because the inversion path must preserve a stable latent trajectory, while the inference path is allowed to use a modern denoising sampler optimized for final generation.

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

This repository’s style transfer has three conceptual stages, regardless of wrapper scripts:

1. Build conditions from content and style inputs.
2. Move content into latent space (inversion) and apply StyleSSP latent edits.
3. Reconstruct with SDXL + ControlNet + IP-Adapter conditions.

In the current batch CLI, stage 3 is executed with `StableDiffusionXLControlNetInpaintPipeline` (from `pipeline_controlnet_inpaint_sd_xl.py`).

#### Core Process Details (Current Path: Inpaint)

1. **Style and content descriptors are extracted**
  * Style image and content image are encoded by IP-Adapter-Instruct into decoupled embeddings (style branch and composition branch).
  * The content text prompt is either user-defined (`--content_image_prompt`) or inferred per image with BLIP.

2. **Content latent is initialized by inversion**
  * `inversion.py` maps the content image into the diffusion latent trajectory.
  * StyleSSP then applies frequency-domain latent manipulation (`freq_exp`) before final reconstruction.

3. **Inpaint pipeline prepares all conditioning tensors**
  * `image`: the content image at target resolution.
  * `mask_image`: full-white mask in this workflow, so the full frame is open to stylization.
  * `control_image`: structure hint(s) from tile/canny/depth (single map or list for multi-control).
  * `ip_adapter_image`: the selected style reference image.
  * `latents`: edited inversion latent (`latent_l`) passed explicitly.

4. **Denoising loop fuses text, control, and style signals**
  * Text guidance uses `guidance_scale` through classifier-free guidance.
  * ControlNet contribution is scaled by `controlnet_conditioning_scale`.
  * Additional style/content steering is injected through `style_guidance_scale` and `content_guidance_scale` (the pipeline `cond_fn` branch computes similarity-driven guidance and adjusts latent updates).
  * Optional `inv_guidance` and `npi_interp` influence latent initialization and negative-prompt interpolation behavior.

5. **Decode and write output**
  * Final latents are decoded by the SDXL VAE and saved as the stylized result.

#### Why Inpaint Is Used In The Batch Script

The batch script currently builds `StableDiffusionXLControlNetInpaintPipeline` explicitly for reconstruction because it offers one place to combine all of the following at once: mask conditioning, one-or-many ControlNet condition images, explicit latent injection, and IP-Adapter style image conditioning.

#### How The Other Two Pipelines Would Work

1. **`StableDiffusionXLImg2ImgPipeline` (from `pipeline_controlnet_sd_xl_img2img.py`)**
  * This is an img2img-style reconstruction path that starts from an input image and adds noise controlled by `strength`.
  * It supports text guidance and IP-Adapter image conditioning in its denoising loop.
  * Compared to inpaint, it does not use a mask branch (`mask_image` / `masked_image_latents`) and therefore has a simpler latent preparation path.
  * In this repository, this pipeline is mainly the inversion/inference companion loaded via `src/utils/enums_utils.py` and `get_pipes(...)`.

2. **`StableDiffusionXLControlNetImg2ImgPipeline` (from `pipeline_controlnet_sd_xl_img2img_plus.py`)**
  * This extends img2img with explicit ControlNet image conditioning (`control_image`) and ControlNet scaling (`controlnet_conditioning_scale`).
  * It includes options like `guess_mode` and supports both single and multi-ControlNet settings.
  * It also keeps style/content guidance hooks (`style_guidance_scale`, `content_guidance_scale`) for additional embedding-driven steering.
  * Conceptually, it sits between plain img2img and inpaint: it has ControlNet conditioning like inpaint, but no mask-specific inpaint branch.

#### Practical Interpretation

* **Inpaint (current):** strongest structural control surface in this repo because mask + ControlNet + explicit latent injection are all active together.
* **Img2img:** simpler transform-from-image route, useful when mask semantics are unnecessary.
* **Img2img_plus:** img2img with stronger structure conditioning through ControlNet, but still without the inpaint mask branch.

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
