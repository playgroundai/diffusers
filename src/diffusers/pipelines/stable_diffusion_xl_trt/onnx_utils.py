import onnxruntime
import numpy as np
from pathlib import Path


def convert_to_onnx_frames(input_dict, device_id=0):
    for input_name in input_dict:
        input_dict[input_name] = onnxruntime.OrtValue.ortvalue_from_numpy(
            convert_to_np_type(input_dict[input_name]), 'cuda',
            device_id=device_id)

    return input_dict


def convert_to_np_type(frames: np.array):
    return frames.cpu().numpy()


def get_onnx_path(model_name, onnx_dir, opt=True):
    onnx_model_dir = Path(onnx_dir) / (model_name + ('.opt' if opt else ''))
    onnx_model_dir.mkdir(exist_ok=True, parents=True)
    return str(onnx_model_dir / 'model.onnx')
