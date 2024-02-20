import torch

class ImageOnlyVaeRunner:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def setup_model(self):
        dtype = self.model.dtype
        self.model.to(dtype=torch.float32)
        self.model.post_quant_conv.to(dtype)
        self.model.decoder.conv_in.to(dtype)
        self.model.decoder.mid_block.to(dtype)
        self.set_latents_to = next(iter(self.model.post_quant_conv.parameters())).dtype

    def run(self, latents, latents_already_scaled = False):
        latents = latents.to(self.set_latents_to)

        with torch.no_grad():
            if not latents_already_scaled:
                latents = latents / self.model.config.scaling_factor

            image = self.model.decode(latents, return_dict=False)[0]

        return image

    @staticmethod
    def _get_latent_sizes(image_height, image_width):
        return image_height // 8, image_width // 8

    def get_sample_input(self, image_height, image_width, batch_size=1):
        latent_height, latent_width = self._get_latent_sizes(image_height, image_width)

        return {'latent': torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)}

    def warmup(self, image_height, image_width, batch_size=1):
        self.run(self.get_sample_input(image_height, image_width, batch_size)['latent'])
