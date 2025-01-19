import torch
# from torch_models.ver3_2_deint import DeintNet
import tensorrt as trt
import onnx
import torch.nn.utils.prune as prune
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50 as FCN_Model

CHOSEN_MODEL = FCN_Model(pretrained=True, progress=False).eval()
ONNX_SAVE_PATH = r"D:\csi4900\onnx-models\deepLabv3_torchVision.onnx"       # Define your own path


def convert_model_to_onnx(pretrained_params_path=None, input_shape=None, dynamic=False, dynamic_shape_range=None, onnx_file_path=""):
    
    # Change this to some sort of ENV variable
    pytorch_model = CHOSEN_MODEL
    if onnx_file_path == "":
        onnx_file_path = ONNX_SAVE_PATH

    if isinstance(input_shape, list):
        dummy_input = [torch.randn(*shape) for shape in input_shape]
        num_inputs = len(input_shape)
    else:
        dummy_input = torch.randn(*input_shape)
        num_inputs = 1
    
    if pretrained_params_path:
        checkpoint = torch.load(pretrained_params_path)
        print("Keys in checkpoint: ", checkpoint.keys())

        # List of possible keys for state dictionaries
        possible_keys = ['model_param', 'state_dict', 'model_state']
        state_dict_key = None

        for key in possible_keys:
            if key in checkpoint:
                state_dict_key = key
                break

        print("state_dict_key:", state_dict_key)
        if state_dict_key:
            pytorch_model.load_state_dict(checkpoint[state_dict_key], strict=False)
            print(f"Loaded pretrained model parameters from {pretrained_params_path} using key '{state_dict_key}'")
        else:
            print("No matching key found in checkpoint for state dictionary")


    # Create dummy output to determine the number of outputs
    with torch.no_grad():
        dummy_output = pytorch_model(*[dummy_input]) if num_inputs > 1 else pytorch_model(dummy_input)
    num_outputs = 1 if not isinstance(dummy_output, tuple) else len(dummy_output)
    
    input_names = [f"input_{i}" for i in range(num_inputs)]
    output_names = [f"output_{i}" for i in range(num_outputs)]

    if dynamic:
        if not dynamic_shape_range:
            min_shape = tuple(map(int, input("Enter min shape as comma-separated values: ").split(',')))
            opt_shape = tuple(map(int, input("Enter opt shape as comma-separated values: ").split(',')))
            max_shape = tuple(map(int, input("Enter max shape as comma-separated values: ").split(',')))
            dynamic_shape_range = {'min': min_shape, 'opt': opt_shape, 'max': max_shape}

        # Dynamic axes for each input and output
        print("input_shape: ", input_shape)

        # Initialize an empty dictionary for dynamic_axes
        dynamic_axes = {}
        for name, shape in zip(input_names, [input_shape]):
            height_index = len(shape) - 2
            width_index = len(shape) - 1
            dynamic_axes[name] = {0: 'batch', height_index: 'height', width_index: 'width'}

        for name in output_names:
            # Assuming that the dynamic dimensions for output are at indices 2 and 3
            dynamic_axes[name] = {0: 'batch_out', 2: 'height_out', 3: 'width_out'}
        print(f"Updated dynamic_axes: {dynamic_axes}")

        torch.onnx.export(pytorch_model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    else:
        try:
            torch.onnx.export(pytorch_model, dummy_input, onnx_file_path)
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")

    model_onnx = onnx.load(onnx_file_path)
    onnx.checker.check_model(model_onnx)

    input_shapes_onnx = [node.type.tensor_type.shape for node in model_onnx.graph.input]
    output_shapes_onnx = [node.type.tensor_type.shape for node in model_onnx.graph.output]
    print(f"Input shapes from ONNX model: {input_shapes_onnx}")
    print(f"Output shapes from ONNX model: {output_shapes_onnx}")

    input_names_onnx = [node.name for node in model_onnx.graph.input]
    output_names_onnx = [node.name for node in model_onnx.graph.output]
    print(f"Input names from ONNX model: {input_names_onnx}")
    print(f"Output names from ONNX model: {output_names_onnx}")
 


# ONNX MODEL CONVERSION DRIVER CODE
if __name__ == '__main__':
    input_shape_str = input("Enter the input shape (N, C, H, W): ")
    input_shape = tuple(map(int, input_shape_str.split(',')))

    dynamic_shapes = input("Do you want to use dynamic shapes? (y/n): ")

    if dynamic_shapes.lower() == 'y':
        min_shape_str = input("Enter the minimum shape (N, C, H, W): ")
        min_shape = tuple(map(int, min_shape_str.split(',')))

        opt_shape_str = input("Enter the optimum shape (N, C, H, W): ")
        opt_shape = tuple(map(int, opt_shape_str.split(',')))

        max_shape_str = input("Enter the maximum shape (N, C, H, W)): ")
        max_shape = tuple(map(int, max_shape_str.split(',')))

        dynamic_shape_range = {
            'min': min_shape,
            'opt': opt_shape,
            'max': max_shape
        }
        dynamic = True
    else:
        dynamic_shape_range = None
        dynamic = False

    onnx_save_path = input("Enter the ONNX save path: ")

    model = FCN_Model(pretrained=True, progress=False).eval()

    convert_model_to_onnx(
        input_shape=input_shape,
        dynamic=dynamic,
        dynamic_shape_range=dynamic_shape_range,
        onnx_file_path=onnx_save_path
    )