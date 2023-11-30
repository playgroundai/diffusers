import os
import torch

from diffusers import AutoencoderKL

from .model_utils import BaseModel


class VAE(BaseModel):
    def __init__(self,
                 version,
                 pipeline,
                 hf_token,
                 device,
                 verbose,
                 max_batch_size,
                 ):
        super(VAE, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose,
                                  max_batch_size=max_batch_size)

    def get_model(self, framework_model_dir):
        vae_decoder_model_path = os.path.join(framework_model_dir, self.version, self.pipeline, "vae_decoder")
        if not os.path.exists(vae_decoder_model_path):
            vae = AutoencoderKL.from_pretrained(self.path,
                                                subfolder="vae",
                                                use_safetensors=self.hf_safetensor,
                                                use_auth_token=self.hf_token).to(self.device)
            vae.save_pretrained(vae_decoder_model_path)
        else:
            print(f"[I] Load VAE decoder pytorch model from: {vae_decoder_model_path}")
            vae = AutoencoderKL.from_pretrained(vae_decoder_model_path).to(self.device)
        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
        return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width),
                       (batch_size, 4, latent_height, latent_width),
                       (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


def make_VAE(version, pipeline, hf_token, device, verbose, max_batch_size):
    return VAE(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size)
