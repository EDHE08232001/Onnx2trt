#pragma once
#ifndef PythonBridge_HPP
#define PythonBridge_HPP
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;


class PythonBridge {
public:
	PythonBridge();
	~PythonBridge();
	void callConvertModelToOnnx(const std::string& pretrainedParamsPath, const std::vector<int>& inputShape,
		bool dynamic, const std::unordered_map<std::string, std::vector<int>>& dynamicShapeRange);

private:
	// py::object onnxConverterModule;
};


#endif