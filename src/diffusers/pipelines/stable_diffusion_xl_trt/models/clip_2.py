import os
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from .clip import CLIP
from .model_utils import get_path, get_clipwithproj_embedding_dim


class CLIPWithProj(CLIP):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        device='cuda',
        verbose=True,
        max_batch_size=16,
        output_hidden_states=False,
        subfolder="text_encoder_2"):

        super(CLIPWithProj, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size, embedding_dim=get_clipwithproj_embedding_dim(version, pipeline), output_hidden_states=output_hidden_states)
        self.subfolder = subfolder

    def get_model(self, framework_model_dir):
        clip_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, "text_encoder_2")
        if not os.path.exists(clip_model_dir):
            model = CLIPTextModelWithProjection.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModelWithProjection.from_pretrained(clip_model_dir).to(self.device)
        return model

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output

def make_CLIPWithProj(version, pipeline, hf_token, device, verbose, max_batch_size, subfolder="text_encoder_2", output_hidden_states=False):
    return CLIPWithProj(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size, subfolder=subfolder, output_hidden_states=output_hidden_states)

def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer"):
    tokenizer_model_dir = os.path.join(framework_model_dir, version, pipeline.name, subfolder)
    if not os.path.exists(tokenizer_model_dir):
        model = CLIPTokenizer.from_pretrained(get_path(version, pipeline),
                subfolder=subfolder,
                use_safetensors=pipeline.is_sd_xl(),
                use_auth_token=hf_token)
        model.save_pretrained(tokenizer_model_dir)
    else:
        print(f"[I] Load tokenizer pytorch model from: {tokenizer_model_dir}")
        model = CLIPTokenizer.from_pretrained(tokenizer_model_dir)
    return model