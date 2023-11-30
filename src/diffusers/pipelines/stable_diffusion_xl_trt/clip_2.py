import onnxruntime
from cuda import cudart
import gc
import nvtx
import onnx
import os
import time
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .clip import CLIPRunner, cache_clip_models
from .models.clip_2 import CLIPWithProj, make_tokenizer
# from aitemplate_stuff.src.compile_lib.compile_clip import compile_clip
from .onnx_utils import convert_to_onnx_frames, get_onnx_path
from .trt_utils import Engine, get_engine_path
from .sd_utils import PIPELINE_TYPE, get_path, timing_decorator


class CLIP2Runner(CLIPRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, make_tokenizer=False)
        self.subfolder = "text_encoder_2"
        self.model_name = "clip2"
        self.tokenizer = make_tokenizer(self.version, self.pipeline_type, self.hf_token, self.framework_model_dir,
                                         subfolder="tokenizer_2")
        self._make_dirs()

    def make_clip_with_proj(self):
        return CLIPWithProj(self.version, self.pipeline_type, self.hf_token, device=self.device, verbose=self.verbose,
                            max_batch_size=self.max_batch_size, output_hidden_states=self.output_hidden_states)

    @timing_decorator
    def run_onnx(self, model, feed_dict):
        ort_feed_dict = convert_to_onnx_frames(feed_dict)
        return model.run(None, ort_feed_dict)

    @timing_decorator
    def run_engine(self, engine, feed_dict):
        outputs = engine.infer(feed_dict, self.stream, use_cuda_graph=False)
        text_embeddings = outputs['text_embeddings'].clone()

        hidden_states = None
        if self.output_hidden_states:
            hidden_states = outputs['hidden_states'].clone()

        return text_embeddings, hidden_states

    @timing_decorator
    def run_pytorch(self, model, tokenized_txt):
        with torch.no_grad():
            outputs = model(tokenized_txt, output_hidden_states=self.output_hidden_states)
            hidden_states = None
            if self.output_hidden_states:
                hidden_states = outputs['hidden_states'][21]

            return outputs["last_hidden_state"], hidden_states

    @timing_decorator
    def run_pytorch_compiled(self, model, tokenized_txt):
        return self.run_pytorch(model, tokenized_txt)


if __name__ == "__main__":
    clip_runner = CLIP2Runner(framework_model_dir="model_caches", output_hidden_states=True)
    clip_obj = clip_runner.make_clip_with_proj()
    clip_model = clip_runner.get_clip_model(clip_obj)
    # compiled_clip_model = torch.compile(clip_model)

    # Cache all models
    cache_clip_models(clip_runner, clip_obj, clip_model)

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    negative_prompt = ""
    pos_tokenized_txt, neg_tokenized_txt = clip_runner.tokenizer_prompts(prompt, negative_prompt, tokenizer=clip_runner.tokenizer)

    # Run with pytorch
    pth_txt_emb, pth_txt_hidden = clip_runner.run_pytorch(clip_model, pos_tokenized_txt)

    # RUn with compiled pytorch
    # pth_comp_txt_emb, pth_comp_txt_hidden = clip_runner.run_pytorch_compiled(compiled_clip_model, pos_tokenized_txt)

    # Run with ONNX CPU

    # Run with ONNX GPU
    # gpu_onnx_model = clip_runner.load_onnx(['CUDAExecutionProvider'], False)
    # onnx_txt_emb = clip_runner.run_onnx(gpu_onnx_model, {"input_ids": pos_tokenized_txt})

    # Run with AITemplate

    # Run with tensorrt
    engine = clip_runner.load_engine(clip_obj, batch_size=1)
    txt_emb, txt_hidden = clip_runner.run_engine(engine, {"input_ids": pos_tokenized_txt})

    assert torch.isclose(pth_txt_emb, txt_emb, atol=0.1).all().item()
    print("")
