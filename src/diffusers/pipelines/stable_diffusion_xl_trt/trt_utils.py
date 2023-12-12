from collections import OrderedDict

from cuda import cudart
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}


def get_engine_path(model_name, engine_dir):
    return os.path.join(engine_dir, model_name + '.plan')

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine():
    def __init__(
            self,
            engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.graph = None  # cuda graph
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build_OWN(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False,
              enable_all_tactics=False, timing_cache=None, update_output_names=None):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        # Create a dynamic shape profile for the "samples" input
        min_shape = (2, 4, 32, 32)  # Minimum shape
        opt_shape = (2, 4, 128, 128)  # Optimal shape
        max_shape = (2, 4, 128, 128)  # Maximum shape

        p.add("samples", min=min_shape, opt=opt_shape, max=max_shape)

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        engine = engine_from_network(
            network,
            config=CreateConfig(fp16=fp16,
                                refittable=enable_refit,
                                profiles=[p],
                                load_timing_cache=timing_cache,
                                **config_kwargs
                                ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False,
              enable_all_tactics=False, timing_cache=None, update_output_names=None):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        engine = engine_from_network(
            network,
            config=CreateConfig(fp16=fp16,
                                refittable=enable_refit,
                                profiles=[p],
                                load_timing_cache=timing_cache,
                                **config_kwargs
                                ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)

            # TODO: Don't wastefully allocate input buffers for things that are always overridden via
            #       `set_tensor` (for loop-invariant inputs that don't need copying). This wastes a few MB
            #        of GPU memory.
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype], device=device)
            self.tensors[binding] = tensor

            # By default, the tensor we allocate here is used for the binding.
            self.context.set_tensor_address(binding, self.tensors[binding].data_ptr())

        # Delete any existing cuda graph, since it will refer to the previous allocations.
        if self.graph is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            CUASSERT(cudart.cudaGraphDestroy(self.graph))
            self.cuda_graph_instance = None
            self.graph = None

    def set_tensor(self, name, buf):
        self.context.set_tensor_address(name, buf.data_ptr())

    # Copy `buf` into the fixed TRT tensor identified by `name`.
    # When using the graph API, the input pointers cannot change between invocations, so all inputs
    # need to be copied. In some cases, the cost of the copy is still less than the cost of not using
    # the graph API. A C++ port can eliminate this silliness.
    def copy_into_tensor(self, name, buf):
        self.tensors[name].copy_(buf)

    # Infer using the cuda graph API. This is more efficient than launching without the graph API (since it launches
    # all the kernels and stuff), but the pointers to the underlying buffers cannot be changed between calls, meaning
    # extra copies often need to be inserted. For that reason, this may be slower than `infer_without_graph`.
    def infer_using_graph(self, stream):
        if self.graph is None:
            # This is very annoying: TensorRT sometimes likes to do a cudaMalloc the first time you run it.
            # That breaks use of the graph API, since you can't do that in record mode. To mitigate this, we have
            # to do this silly lazy-init behaviour: the first time we're invoked we infer _first_, then do a recording
            # to build the graph, so the second and subsequent times use the graph. Silly!
            self.context.execute_async_v3(stream)

            # capture cuda graph
            CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
            self.context.execute_async_v3(stream)
            self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
            self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 1)) # cudaGraphInstantiateFlagAutoFreeOnLaunch
        else:
            CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))

        return self.tensors

    def infer_without_graph(self, stream):
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")
        return self.tensors

    # TODO: eliminate this flawed function :D
    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(stream)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors