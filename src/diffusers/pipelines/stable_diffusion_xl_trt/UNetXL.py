from cuda import cudart
from .trt_utils import Engine, get_engine_path
from .sd_utils import PIPELINE_TYPE
from os import path
import torch

def get_unet_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0") and pipeline.is_sd_xl_base():
        return 2048
    elif version in ("xl-1.0") and pipeline.is_sd_xl_refiner():
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

class UNETXLRunnerInfer:
    def __init__(self, version='xl-1.0', pipeline_type=PIPELINE_TYPE.SD_XL_BASE, hf_token=None, scheduler=None,
                 vae_scaling_factor=0.18215, fp16=True, max_batch_size=1, device='cuda',
                 framework_model_dir='pytorch_model', verbose=True, guidance_scale=5.0, guidance_rescale=0.0,
                 stream=None):
        self.version = version
        self.pipeline_type = pipeline_type
        self.hf_token = hf_token
        self.nvtx_profile = None
        self.device = device
        self.verbose = verbose
        self.scheduler = scheduler
        self.vae_scaling_factor = vae_scaling_factor
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.framework_model_dir = framework_model_dir
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.model_name = "unetxl"
        self.subfolder = "unetxl"
        self.onnx_opset = 17
        self.unet_dim = 4
        self.time_dim = 5 if self.pipeline_type.is_sd_xl_refiner() else 6
        self.embedding_dim = get_unet_embedding_dim(self.version, self.pipeline_type)
        self.text_maxlen=77
        self.stream = stream
        self.trt_dir = path.join(self.framework_model_dir, self.version, self.pipeline_type.name, self.subfolder,
                                    "TRT", "VARIABLE_SHAPE")

    @staticmethod
    def _get_latent_sizes(image_height, image_width):
        return image_height // 8, image_width // 8

    def get_shape_dict(self, image_height, image_width, batch_size=1):
        latent_height, latent_width = self._get_latent_sizes(image_height, image_width) 

        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width),
            'text_embeds': (2*batch_size, 1280),
            'time_ids': (2*batch_size, self.time_dim)
        }

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

    def load_engine(self):
        engine = Engine(get_engine_path(self.model_name, self.trt_dir))
        engine.load()
        self.activate_engines(engine)

        return engine

    # @timing_decorator
    def run_engine(self, engine, feed_dict):
        outputs = engine.infer(feed_dict, self.stream, use_cuda_graph=False)
        return outputs

    def warmup(self, engine, image_width, image_height, batch_size=1):
        engine.allocate_buffers(shape_dict=self.get_shape_dict(image_height, image_width), device=self.device)

        engine.infer_using_graph(self.stream)

    def get_sample_input(self, image_height, image_width, batch_size=1):
        dtype = torch.float16 if self.fp16 else torch.float32
        latent_height, latent_width = self._get_latent_sizes(image_height, image_width) 
        
        return {
            'sample': torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            'timestep': torch.tensor([1.], dtype=torch.float32, device=self.device),
            'encoder_hidden_states': torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            'text_embeds': torch.randn(2*batch_size, 1280, dtype=dtype, device=self.device),
            'time_ids': torch.randn(2*batch_size, self.time_dim, dtype=dtype, device=self.device)
        }
