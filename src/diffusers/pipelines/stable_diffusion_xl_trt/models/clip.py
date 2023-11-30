from .model_utils import BaseModel, Optimizer, get_clip_embedding_dim

from pathlib import Path
import torch
from transformers import CLIPTextModel


class CLIP(BaseModel):
    def __init__(self,
                 version,
                 pipeline,
                 hf_token,
                 device,
                 verbose,
                 max_batch_size,
                 embedding_dim,
                 output_hidden_states=False,
                 subfolder="text_encoder"
                 ):
        super(CLIP, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose,
                                   max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.subfolder = subfolder

        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ['hidden_states']

    def get_model(self, framework_model_dir):
        clip_model_dir = Path(framework_model_dir) / self.version / self.pipeline / "text_encoder"
        if not clip_model_dir.exists():
            model = CLIPTextModel.from_pretrained(self.path,
                                                  subfolder=self.subfolder,
                                                  use_safetensors=self.hf_safetensor,
                                                  use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModel.from_pretrained(clip_model_dir).to(self.device)
        return model

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
        return ['text_embeddings']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width,
                                                                            static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings'])  # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        if 'hidden_states' in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(return_onnx=True)
            opt.info(self.name + ': added hidden_states')
        opt.info(self.name + ': finished')
        return opt_onnx_graph


def make_CLIP(version, pipeline, hf_token, device, verbose, max_batch_size, output_hidden_states=False,
              subfolder="text_encoder"):
    return CLIP(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size,
                embedding_dim=get_clip_embedding_dim(version, pipeline), output_hidden_states=output_hidden_states,
                subfolder=subfolder)
