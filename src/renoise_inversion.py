# Copyright (c) 2023 pix2pixzero
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
from torchvision import transforms

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


def spherical_dist_loss(x, y):
    # Ensure both are [N, D]
    if x.ndim == 2 and y.ndim == 2 and x.shape[0] != y.shape[0]:
        # broadcast pooled embedding to match token count
        if x.shape[0] == 1:
            x = x.expand(y.shape[0], -1)
        elif y.shape[0] == 1:
            y = y.expand(x.shape[0], -1)

    return -(x @ y.T)



@torch.no_grad()
def rescale_guidance(guidance, noise_pred_text, noise_pred_uncond, guidance_scale, cutoff=2000.0):
    norm_cfg = torch.norm(guidance_scale * (noise_pred_text - noise_pred_uncond), p=2)
    norm_guidance = torch.norm(guidance, p=2)
    scale = norm_cfg / norm_guidance
    scale = torch.where(
        scale < cutoff,
        scale,
        torch.tensor(cutoff, device=scale.device),
    )
    return scale


@torch.enable_grad()
def get_guidace(
    pipe_inf,
    latents,
    prompt,
    feature_extractor,
    style_embedding,
    content_embedding,
    neg_style_embedding,
    neg_content_embedding,
    clip_model_used=True,
    get_grad_guidance=False,
):
    origin_dtype = latents.dtype

    if get_grad_guidance:
        latents = latents.detach().requires_grad_(True)

        pipe_inf.vae.to(dtype=torch.float32)
        latents_fp32 = latents.to(dtype=torch.float32)
        latents_fp32 = latents_fp32 / pipe_inf.vae.config.scaling_factor
        image = pipe_inf.vae.decode(latents_fp32, return_dict=False)[0]

        if clip_model_used:
            _, content_output, style_output = feature_extractor(
                normalize(transforms.Resize(224)(image[0:1]))
            )
        else:
            image_tensor = transforms.Resize(224)(image)
            clip_image = image_tensor.to(pipe_inf.device)
            style_output = feature_extractor.get_decouple_embeds(
                clip_image=clip_image, prompt="", query="use the style from the image"
            ).to(latents.dtype)
            content_output = feature_extractor.get_decouple_embeds(
                clip_image=clip_image, prompt="", query="use the composition from the image"
            ).to(latents.dtype)
    else:
        img = pipe_inf(
            prompt=prompt,
            num_inference_steps=1,
            negative_prompt=prompt,
            image=latents,
            strength=pipe_inf.cfg.inversion_max_step,
            denoising_start=1.0 - pipe_inf.cfg.inversion_max_step,
            guidance_scale=1.0,
            get_grad_guidance=get_grad_guidance,
        ).images[0]

        if clip_model_used:
            _, content_output, style_output = feature_extractor(
                normalize(transforms.Resize(224)(img[None]))
            )
        else:
            style_output = feature_extractor.get_decouple_embeds(
                pil_image=img, prompt="", query="use the style from the image"
            )
            content_output = feature_extractor.get_decouple_embeds(
                pil_image=img, prompt="", query="use the composition from the image"
            )

    loss = 0.0

    if pipe_inf.cfg.inv_style_guidance_scale > 0.0:
        style_loss = spherical_dist_loss(
            style_output, style_embedding.to(latents.dtype)
        ).mean() * pipe_inf.cfg.inv_style_guidance_scale
        loss += style_loss

    if pipe_inf.cfg.inv_content_guidance_scale > 0.0:
        content_loss = spherical_dist_loss(
            content_output, content_embedding.to(latents.dtype)
        ).mean() * pipe_inf.cfg.inv_content_guidance_scale
        loss += content_loss

    if pipe_inf.cfg.inv_neg_style_guidance_scale > 0.0:
        neg_style_loss = -spherical_dist_loss(
            style_output, neg_style_embedding.to(latents.dtype)
        ).mean() * pipe_inf.cfg.inv_neg_style_guidance_scale
        loss += neg_style_loss

    if pipe_inf.cfg.inv_neg_content_guidance_scale > 0.0:
        neg_content_loss = -spherical_dist_loss(
            content_output, neg_content_embedding.to(latents.dtype)
        ).mean() * pipe_inf.cfg.inv_neg_content_guidance_scale
        loss += neg_content_loss

    if get_grad_guidance:
        grad = -torch.autograd.grad(loss, latents)[0]
        latents = latents.to(origin_dtype)
        grad = grad.to(origin_dtype)
        pipe_inf.vae.to(dtype=torch.float16)
        return grad

    return loss


@torch.no_grad()
def from_latents2img(pipe_inf, latents):
    needs_upcasting = pipe_inf.vae.dtype == torch.float16 and pipe_inf.vae.config.force_upcast
    if needs_upcasting:
        pipe_inf.upcast_vae()
        latents = latents.to(next(iter(pipe_inf.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe_inf.vae.dtype:
        if torch.backends.mps.is_available():
            pipe_inf.vae = pipe_inf.vae.to(latents.dtype)

    has_latents_mean = hasattr(pipe_inf.vae.config, "latents_mean") and pipe_inf.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe_inf.vae.config, "latents_std") and pipe_inf.vae.config.latents_std is not None

    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe_inf.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe_inf.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / pipe_inf.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe_inf.vae.config.scaling_factor

    image = pipe_inf.vae.decode(latents, return_dict=False)[0]

    if needs_upcasting:
        pipe_inf.vae.to(dtype=torch.float16)

    return image


@torch.no_grad()
def unet_pass(pipe, z_t, t, prompt_embeds, added_cond_kwargs):
    latent_model_input = torch.cat([z_t] * 2) if pipe.do_classifier_free_guidance else z_t
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    return pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]


def inversion_step(
    pipe,
    z_t: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    num_renoise_steps: int = 100,
    first_step_max_timestep: int = 250,
    generator=None,
    pipe_inf=None,
    prompt=None,
    feature_extractor=None,
    style_embedding=None,
    content_embedding=None,
    neg_style_embedding=None,
    neg_content_embedding=None,
    enable_guidance: bool = False,
    used_NPI_guidance: bool = False,
) -> torch.Tensor:
    extra_step_kwargs = {}

    avg_range = (
        pipe.cfg.average_first_step_range
        if t.item() < first_step_max_timestep
        else pipe.cfg.average_step_range
    )
    num_renoise_steps = (
        min(pipe.cfg.max_num_renoise_steps_first_step, num_renoise_steps)
        if t.item() < first_step_max_timestep
        else num_renoise_steps
    )

    nosie_pred_avg = None
    z_tp1_forward = pipe.scheduler.add_noise(pipe.z_0, pipe.noise, t.view((1))).detach()
    approximated_z_tp1 = z_t.clone()

    for i in range(num_renoise_steps + 1):
        with torch.no_grad():
            prompt_embeds_in = prompt_embeds
            added_cond_kwargs_in = added_cond_kwargs

            noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds_in, added_cond_kwargs_in)

            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if i >= avg_range[0] and i < avg_range[1]:
                j = i - avg_range[0]
                if nosie_pred_avg is None:
                    nosie_pred_avg = noise_pred.clone()
                else:
                    nosie_pred_avg = j * nosie_pred_avg / (j + 1) + noise_pred / (j + 1)

        step_out = pipe.scheduler.step(
            noise_pred,
            t,
            approximated_z_tp1,
            **extra_step_kwargs,
            return_dict=False,
        )
        approximated_z_tp1 = step_out[0].detach()

        if enable_guidance and pipe_inf is not None and feature_extractor is not None:
            guidance = get_guidace(
                pipe_inf=pipe_inf,
                latents=approximated_z_tp1,
                prompt=prompt,
                feature_extractor=feature_extractor,
                style_embedding=style_embedding,
                content_embedding=content_embedding,
                neg_style_embedding=neg_style_embedding,
                neg_content_embedding=neg_content_embedding,
                clip_model_used=False,
                get_grad_guidance=pipe.cfg.get_grad_guidance,
            )

            if pipe.do_classifier_free_guidance:
                _, noise_pred_text = noise_pred.chunk(2)
                scale = rescale_guidance(guidance, noise_pred_text, noise_pred_uncond, pipe.guidance_scale)
                guidance = guidance * scale

            guided_noise = noise_pred - guidance
            step_out = pipe.scheduler.step(
                guided_noise,
                t,
                approximated_z_tp1,
                **extra_step_kwargs,
                return_dict=False,
            )
            approximated_z_tp1 = step_out[0].detach()

    if pipe.cfg.average_latent_estimations and nosie_pred_avg is not None:
        step_out = pipe.scheduler.step(
            nosie_pred_avg,
            t,
            z_t,
            **extra_step_kwargs,
            return_dict=False,
        )
        approximated_z_tp1 = step_out[0].detach()

    return approximated_z_tp1
