from enum import auto, Enum
import numpy as np
import time
import torch

# from schedulers import DDIMScheduler, DPMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, \
#     PNDMScheduler, UniPCMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler

from .constants import lora_storage_dir, storage_dir

def get_storage_dir(use_lora: bool = False):
    return lora_storage_dir if use_lora else storage_dir


class PIPELINE_TYPE(Enum):
    TXT2IMG = auto()
    IMG2IMG = auto()
    INPAINT = auto()
    SD_XL_BASE = auto()
    SD_XL_REFINER = auto()

    def is_txt2img(self):
        return self == self.TXT2IMG

    def is_img2img(self):
        return self == self.IMG2IMG

    def is_inpaint(self):
        return self == self.INPAINT

    def is_sd_xl_base(self):
        return self == self.SD_XL_BASE

    def is_sd_xl_refiner(self):
        return self == self.SD_XL_REFINER

    def is_sd_xl(self):
        return self.is_sd_xl_base() or self.is_sd_xl_refiner()


def get_path(version, pipeline, controlnet=None):
    if controlnet is not None:
        return ["lllyasviel/sd-controlnet-" + modality for modality in controlnet]

    if version == "1.4":
        if pipeline.is_inpaint():
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        if pipeline.is_inpaint():
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "runwayml/stable-diffusion-v1-5"
    elif version == "2.0-base":
        if pipeline.is_inpaint():
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        if pipeline.is_inpaint():
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    elif version == 'xl-1.0':
        if pipeline.is_sd_xl_base():
            return "stabilityai/stable-diffusion-xl-base-1.0"
        elif pipeline.is_sd_xl_refiner():
            return "stabilityai/stable-diffusion-xl-refiner-1.0"
        else:
            raise ValueError(f"Unsupported SDXL 1.0 pipeline {pipeline.name}")
    else:
        raise ValueError(f"Incorrect version {version}")


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n{func.__name__} took {elapsed_time:.5f}s to execute.\n")
        return result

    return wrapper


def  _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def get_scheduler(scheduler, device, version):
    from schedulers import DDIMScheduler, DPMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, \
        PNDMScheduler, UniPCMultistepScheduler
    # Schedule options
    sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
    if version in ("2.0", "2.1"):
        sched_opts['prediction_type'] = 'v_prediction'
    else:
        sched_opts['prediction_type'] = 'epsilon'
    if scheduler == "DDIM":
        scheduler = DDIMScheduler(device=device, **sched_opts)
    elif scheduler == "DPM":
        scheduler = DPMScheduler(device=device, **sched_opts)
    elif scheduler == "EulerA":
        scheduler = EulerAncestralDiscreteScheduler(device=device, **sched_opts)
    elif scheduler == "LMSD":
        scheduler = LMSDiscreteScheduler(device=device, **sched_opts)
    elif scheduler == "PNDM":
        sched_opts["steps_offset"] = 1
        scheduler = PNDMScheduler(device=device, **sched_opts)
    elif scheduler == "UniPCMultistepScheduler":
        scheduler = UniPCMultistepScheduler(device=device)
    else:
        raise ValueError(f"Scheduler should be either DDIM, DPM, EulerA, LMSD or PNDM")

    return scheduler

def get_scheduler_orig(scheduler, device, version):
    # Schedule options
    from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler

    sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
    # sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012,
    #                     'beta_schedule': 'scaled_linear', 'trained_betas': None,
    #                     'interpolation_type': 'linear', 'use_karras_sigmas': False, 'timestep_spacing': 'leading',
    #                     'steps_offset': 1,
    #                     '_diffusers_version': '0.19.0.dev0', 'clip_sample': False, 'sample_max_value': 1.0,
    #                     'set_alpha_to_one': False, 'skip_prk_steps': True, '_FrozenDict__frozen': True}
    if version in ("2.0", "2.1"):
        sched_opts['prediction_type'] = 'v_prediction'
    else:
        sched_opts['prediction_type'] = 'epsilon'
    if scheduler == "DDIM":
        scheduler = DDIMScheduler(**sched_opts)
    elif scheduler == "DPM":
        scheduler = DPMScheduler(device=device, **sched_opts)
    elif scheduler == "EulerA":
        scheduler = EulerAncestralDiscreteScheduler(**sched_opts)
    elif scheduler == "LMSD":
        scheduler = LMSDiscreteScheduler(device=device, **sched_opts)
    elif scheduler == "PNDM":
        sched_opts["steps_offset"] = 1
        scheduler = PNDMScheduler(device=device, **sched_opts)
    elif scheduler == "UniPCMultistepScheduler":
        scheduler = UniPCMultistepScheduler(device=device)
    else:
        raise ValueError(f"Scheduler should be either DDIM, DPM, EulerA, LMSD or PNDM")

    return scheduler

def get_scheduler_own(scheduler, device, version):
    # Schedule options
    from own_schedulers.DDIM import DDIMScheduler
    # Schedule options
    sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, "beta_schedule": "scaled_linear"}
    if version in ("2.0", "2.1"):
        sched_opts['prediction_type'] = 'v_prediction'
    else:
        sched_opts['prediction_type'] = 'epsilon'
    if scheduler == "DDIM":
        scheduler = DDIMScheduler(device=device, **sched_opts)
    elif scheduler == "DPM":
        scheduler = DPMScheduler(device=device, **sched_opts)
    elif scheduler == "EulerA":
        scheduler = EulerAncestralDiscreteScheduler(device=device, **sched_opts)
    elif scheduler == "LMSD":
        scheduler = LMSDiscreteScheduler(device=device, **sched_opts)
    elif scheduler == "PNDM":
        sched_opts["steps_offset"] = 1
        scheduler = PNDMScheduler(device=device, **sched_opts)
    elif scheduler == "UniPCMultistepScheduler":
        scheduler = UniPCMultistepScheduler(device=device)
    else:
        raise ValueError(f"Scheduler should be either DDIM, DPM, EulerA, LMSD or PNDM")

    return scheduler


def initialize_latents(scheduler, device, generator, batch_size, unet_channels, latent_height, latent_width):
    latents_dtype = torch.float32 # text_embeddings.dtype
    latents_shape = (batch_size, unet_channels, latent_height, latent_width)
    latents = torch.randn(latents_shape, device=device, dtype=latents_dtype, generator=generator)
    # Scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents

def denoise_latent(scheduler,
                   latents,
                   text_embeddings,
                   denoiser='unet',
                   timesteps=None,
                   step_offset=0,
                   mask=None,
                   masked_image_latents=None,
                   guidance=7.5,
                   image_guidance=1.5,
                   add_kwargs={},
                   controlnet_imgs=None,
                   controlnet_scales=None,
                   vae_scaling_factor=0.13025):

    assert guidance > 1.0, "Guidance has to be > 1.0"
    assert image_guidance > 1.0, "Image guidance has to be > 1.0"

    controlnet_imgs = preprocess_controlnet_images(latents.shape[0], controlnet_imgs)

    # cudart.cudaEventRecord(self.events['denoise-start'], 0)
    if not isinstance(timesteps, torch.Tensor):
        timesteps = scheduler.timesteps
    for step_index, timestep in enumerate(timesteps):
        # if self.nvtx_profile:
        #     nvtx_latent_scale = nvtx.start_range(message='latent_scale', color='pink')

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        if controlnet_imgs is None:
            latent_model_input = scheduler.scale_model_input(latent_model_input, step_offset + step_index,
                                                                  timestep)
        else:
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        if isinstance(mask, torch.Tensor):
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
        # if self.nvtx_profile:
        #     nvtx.end_range(nvtx_latent_scale)

        # Predict the noise residual
        # if self.nvtx_profile:
        #     nvtx_unet = nvtx.start_range(message='unet', color='blue')

        timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

        sample_inp = latent_model_input
        timestep_inp = timestep_float
        embeddings_inp = text_embeddings

        params = {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp}
        params.update(add_kwargs)
        if controlnet_imgs is not None:
            params.update({"images": controlnet_imgs, "controlnet_scales": controlnet_scales})

        # TODO THIS NEEDS TO BE CHANGED TO BE PYTORCH UNET FOR LOCAL MACHINE, IDEALLY WE GIVE IT A FUNCTION BY SHOULD JUST PUT THIS ALL IN A UNET CLASS
        noise_pred = self.runEngine(denoiser, params)['latent']

        # if self.nvtx_profile:
        #     nvtx.end_range(nvtx_unet)
        #
        # if self.nvtx_profile:
        #     nvtx_latent_step = nvtx.start_range(message='latent_step', color='pink')

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

        if type(scheduler) == UniPCMultistepScheduler:
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        else:
            latents = scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

        # if self.nvtx_profile:
        #     nvtx.end_range(nvtx_latent_step)

    latents = 1. / vae_scaling_factor * latents
    # cudart.cudaEventRecord(self.events['denoise-stop'], 0)
    return latents

def preprocess_controlnet_images(self, batch_size, images=None):
    '''
    images: List of PIL.Image.Image
    '''
    if images is None:
        return None

    # if self.nvtx_profile:
    #     nvtx_image_preprocess = nvtx.start_range(message='image_preprocess', color='pink')
    images = [
        (np.array(i.convert("RGB")).astype(np.float32) / 255.0)[..., None].transpose(3, 2, 0, 1).repeat(batch_size,
                                                                                                        axis=0) for
        i in images]
    # do_classifier_free_guidance
    images = [torch.cat([torch.from_numpy(i).to(self.device).float()] * 2) for i in images]
    images = torch.cat([image[None, ...] for image in images], dim=0)

    # if self.nvtx_profile:
    #     nvtx.end_range(nvtx_image_preprocess)
    return images