#include "PythonBridge.hpp"

PythonBridge::PythonBridge() {
	Py_Initialize();
	py::module::import("sys").attr("path").attr("append")("D:/csi4900/onnx2trtHawk/onnx-to-trt-old");		// Define your own path
}

PythonBridge::~PythonBridge() {
	Py_Finalize();
}

void PythonBridge::callConvertModelToOnnx(const std::string& pretrainedParamsPath, const std::vector<int>& inputShape,
	bool dynamic, const std::unordered_map<std::string, std::vector<int>>& dynamicShapeRange) {
	try {
		py::module onnxConverterModule = py::module::import("onnx_converter");
		py::dict dynamicShapeRangeDict;
		for (const auto& pair : dynamicShapeRange) {
			dynamicShapeRangeDict[pair.first.c_str()] = py::cast(pair.second);
		}
		onnxConverterModule.attr("convert_model_to_onnx")(pretrainedParamsPath,
			py::cast(inputShape), dynamic, dynamicShapeRangeDict);
	}
	catch (const py::error_already_set& e) {
		std::cerr << "Error calling convert_model_to_onnx: " << e.what() << std::endl;
	}
}