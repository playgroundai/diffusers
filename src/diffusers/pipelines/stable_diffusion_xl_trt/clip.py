import onnxruntime
from cuda import cudart
import gc
import onnx
from os import path
import time
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .models.clip import CLIP
from .onnx_utils import convert_to_onnx_frames, get_onnx_path
from .trt_utils import Engine, get_engine_path
from .sd_utils import PIPELINE_TYPE, get_path, timing_decorator


class CLIPRunner:
    def __init__(self, version='xl-1.0', pipeline_type=PIPELINE_TYPE.SD_XL_BASE, hf_token=None,
                 framework_model_dir='pytorch_model', device='cuda', verbose=False, max_batch_size=1,
                 output_hidden_states=True, stream=None, make_tokenizer=True):
        self.version = version
        self.pipeline_type = pipeline_type
        self.hf_token = hf_token
        self.framework_model_dir = framework_model_dir
        self.device = device
        self.verbose = verbose
        self.max_batch_size = max_batch_size
        self.output_hidden_states = output_hidden_states
        self.opt_image_height = 1024
        self.opt_image_width = 1024
        self.opt_batch_size = 1
        self.onnx_opset = 17
        self.subfolder = "text_encoder"
        if make_tokenizer:
            self.tokenizer = self.make_tokenizer(self.version, self.pipeline_type, self.hf_token, framework_model_dir)
        self.clip_embedding_dim = self.get_clip_embedding_dim()
        self._make_dirs()
        self.model_name = "clip"
        self.stream = stream

    def _make_dirs(self):
        self.onnx_dir = path.join(self.framework_model_dir, self.version, self.pipeline_type.name, self.subfolder,
                                     "ONNX", f"hidden_states_{self.output_hidden_states}")
        self.trt_dir = path.join(self.framework_model_dir, self.version, self.pipeline_type.name, self.subfolder,
                                    "TRT", f"hidden_states_{self.output_hidden_states}")

    @staticmethod
    def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer"):
        tokenizer_model_dir = path.join(framework_model_dir, version, pipeline.name, subfolder)
        if not path.exists(tokenizer_model_dir):
            model = CLIPTokenizer.from_pretrained(get_path(version, pipeline),
                                                  subfolder=subfolder,
                                                  use_safetensors=pipeline.is_sd_xl(),
                                                  use_auth_token=hf_token)
            model.save_pretrained(tokenizer_model_dir)
        else:
            # print(f"[I] Load tokenizer pytorch model from: {tokenizer_model_dir}")
            model = CLIPTokenizer.from_pretrained(tokenizer_model_dir)
        return model

    def make_clip(self):
        return CLIP(self.version, self.pipeline_type, self.hf_token, device=self.device, verbose=self.verbose,
                    max_batch_size=self.max_batch_size, embedding_dim=self.clip_embedding_dim,
                    output_hidden_states=self.output_hidden_states, subfolder=self.subfolder)

    def get_clip_model(self, clip_obj):
        return clip_obj.get_model(self.framework_model_dir)

    def get_clip_embedding_dim(self):
        if self.version in ("1.4", "1.5"):
            return 768
        elif self.version in ("2.0", "2.0-base", "2.1", "2.1-base"):
            return 1024
        elif self.version in ("xl-1.0") and self.pipeline_type.is_sd_xl_base():
            return 768
        elif self.version in ("xl-1.0") and self.pipeline_type.is_sd_xl_refiner():
            return 1280
        else:
            raise ValueError(f"Invalid version {self.version} + pipeline {self.pipeline_type}")

    def cache_trt_model(self, obj, model, force_export=False, force_optimize=False,
                        static_batch=False, static_shape=True, enable_refit=False, enable_preview=False,
                        enable_all_tactics=False, timing_cache=None):
        # Build TensorRT engines
        engine_path = get_engine_path(self.model_name, self.trt_dir)
        self.cache_onnx_model(obj, model, force_export, force_optimize)
        engine = Engine(engine_path)
        onnx_opt_path = get_onnx_path(self.model_name, self.onnx_dir)

        if force_export or not path.exists(engine.engine_path):
            update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
            engine.build(onnx_opt_path,
                         fp16=True,
                         input_profile=obj.get_input_profile(
                             self.opt_batch_size, self.opt_image_height, self.opt_image_width,
                             static_batch=static_batch, static_shape=static_shape
                         ),
                         enable_refit=enable_refit,
                         enable_preview=enable_preview,
                         enable_all_tactics=enable_all_tactics,
                         timing_cache=timing_cache,
                         update_output_names=update_output_names)

        print(f"COMPLETED CACHE OF {engine_path}")

    def cache_onnx_model(self, obj, model, force_export=False, force_optimize=False):
        # Export model to ONNX
        onnx_path = get_onnx_path(self.model_name, self.onnx_dir, opt=False)
        onnx_opt_path = get_onnx_path(self.model_name, self.onnx_dir)
        if force_export or not path.exists(onnx_opt_path):
            if force_export or not path.exists(onnx_path):
                print(f"Exporting model: {onnx_path}")
                with torch.inference_mode(), torch.autocast("cuda"):
                    inputs = obj.get_sample_input(1, self.opt_image_height, self.opt_image_width)
                    output_names = obj.get_output_names()
                    torch.onnx.export(model,
                                      inputs,
                                      onnx_path,
                                      export_params=True,
                                      opset_version=self.onnx_opset,
                                      do_constant_folding=True,
                                      input_names=obj.get_input_names(),
                                      # output_names=obj.get_output_names(),
                                      output_names=output_names,
                                      dynamic_axes=obj.get_dynamic_axes(),
                                      )
                del model
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"Found cached model: {onnx_path}")

            # Optimize onnx
            if force_optimize or not path.exists(onnx_opt_path):
                print(f"Generating optimizing model: {onnx_opt_path}")
                onnx_opt_graph = obj.optimize(onnx.load(onnx_path))
                if onnx_opt_graph.ByteSize() > 2147483648:
                    onnx.save_model(
                        onnx_opt_graph,
                        onnx_opt_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        convert_attribute=False)
                else:
                    onnx.save(onnx_opt_graph, onnx_opt_path)
            else:
                print(f"Found cached optimized model: {onnx_opt_path} ")

    @staticmethod
    def calculate_max_device_memory(engine):
        max_device_memory = 0
        max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activate_engines(self, engine, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculate_max_device_memory(engine)
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        engine.activate(reuse_device_memory=self.shared_device_memory)

    def load_engine(self, obj, batch_size=1):
        engine = Engine(get_engine_path(self.model_name, self.trt_dir))
        engine.load()
        self.activate_engines(engine)
        engine.allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, self.opt_image_height, self.opt_image_width),
                                device=self.device)

        return engine

    def load_onnx(self, providers=None, opt=False):
        if self.output_hidden_states:
            raise NotImplementedError("Have no implemented hidden outputs for ONNX yet")
        if not providers:
            providers = ['CUDAExecutionProvider']
        return onnxruntime.InferenceSession(str(get_onnx_path(self.model_name, self.onnx_dir, opt=opt)),
                                            providers=providers)

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
                hidden_states = outputs['hidden_states'][10]

            return outputs["last_hidden_state"], hidden_states

    @timing_decorator
    def tokenizer_prompts(self, prompt, negative_prompt, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer

        # Tokenize prompt
        text_input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        # Tokenize negative prompt
        uncond_input_ids = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        return text_input_ids, uncond_input_ids

    def get_feed_dict(self, obj, height, width):
        return dict(zip(obj.get_input_names(), obj.get_sample_input(self.max_batch_size, height, width)))

    def warmup(self, engine, obj):
        feed_dict = self.get_feed_dict(obj, self.opt_image_height, self.opt_image_width)
        engine.infer(feed_dict, self.stream, use_cuda_graph=False)


def cache_clip_models(runner, obj, model):
    runner.cache_onnx_model(obj, model)
    runner.cache_trt_model(obj, model)


if __name__ == "__main__":
    clip_runner = CLIPRunner(framework_model_dir="model_caches", output_hidden_states=True)
    clip_obj = clip_runner.make_clip()
    clip_model = clip_runner.get_clip_model(clip_obj)
    # compiled_clip_model = torch.compile(clip_model)

    # Cache all models
    cache_clip_models(clip_runner, clip_obj, clip_model)

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    negative_prompt = ""
    pos_tokenized_txt, neg_tokenized_txt = clip_runner.tokenizer_prompts(prompt, negative_prompt)

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
