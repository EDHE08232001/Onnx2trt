#pragma once
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cctype>
#include <stdlib.h>
#include <numeric>
#include <iterator>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/quality.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "ImageProcessingUtil.hpp"
#include "TRTGeneration.hpp"
#include "TRTInference.hpp"
//#include "PythonBridge.hpp"
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;


// overloading operator for nvinfer1::Dims4
std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims) {
	os << "(";
	for (int i = 0; i < dims.nbDims; ++i) {
		os << dims.d[i];
		if (i < dims.nbDims - 1) {
			os << ", ";
		}
	}
	os << ")";
	return os;
}


int main() {
	std::cout << "current dir: " << filesystem::current_path() << std::endl;

	_putenv_s("CUDA_MODULE_LOADING", "LAZY");
	const char* lazyLoading = getenv("CUDA_MODULE_LOADING");

	// Just leanve it as it is, unless you will choose option other than 3
	/////////////////////////////////////////////////////////////////////////////////////////////////
	string superres_img_path = "../../input_imgs/Anchors.ppm";
	string og_superres_img_path = "../../input_imgs/1080/viet-nam.jpg";
	string seg_img_path = "../../input_imgs/seg_car_img_640-360.png";
	string deeplab_onnxModelPath = "../../ONNX-models/deepLabv3_torchVision_static_4batch.onnx";
	string planFilePath = "../../TRT-plans/mobileone_s4.Hawk2.plan";
	vector<string> img_paths;
	string userInput;
	int option;
	/////////////////////////////////////////////////////////////////////////////////////////////////

	// Choose 3 for ONNX Conversion
	cout << "Choose an option:" << endl;
	cout << "1) PyTorch to TRT conversion" << endl;
	cout << "2) PyTorch to ONNX conversion" << endl;
	cout << "3) ONNX to TRT conversion" << endl;
	cout << "4) Only run inference" << endl;
	cout << "Enter your choice (1-4): ";
	cin >> option;

	switch (option) {
	case 1: {
		// PyTorch to TRT conversion
		string pretrainedParamsPath;
		cout << "Do you want to enter a path for pretrained parameters? (yes/no): ";
		cin >> userInput;
		if (userInput == "yes" || userInput == "y") {
			cout << "Enter the path to the pretrained parameters: ";
			cin >> pretrainedParamsPath;
		}
		else {
			pretrainedParamsPath = "None";
		}

		vector<int> inputShape;
		cout << "Enter the desired input shape (format: N C H W): ";
		int dim;
		for (int i = 0; i < 4; ++i) {
			cin >> dim;
			inputShape.push_back(dim);
		}

		bool dynamic;
		cout << "Do you intend to have dynamic shapes in this model? (yes/no): ";
		cin >> userInput;
		dynamic = (userInput == "yes" || userInput == "y");

		vector<int> minShape, optShape, maxShape;

		unordered_map<string, vector<int>> dynamicShapeRange;
		if (dynamic) {
			cout << "Enter the minimum shape (format: N C H W): ";
			for (int i = 0; i < 4; ++i) {
				cin >> dim;
				minShape.push_back(dim);
			}
			cout << "Enter the optimum shape (format: N C H W): ";
			for (int i = 0; i < 4; ++i) {
				cin >> dim;
				optShape.push_back(dim);
			}
			cout << "Enter the maximum shape (format: N C H W): ";
			for (int i = 0; i < 4; ++i) {
				cin >> dim;
				maxShape.push_back(dim);
			}
			dynamicShapeRange["min"] = minShape;
			dynamicShapeRange["opt"] = optShape;
			dynamicShapeRange["max"] = maxShape;
		}

		string onnxFilePath;
		cout << "Enter the path to save the ONNX model: ";
		cin >> onnxFilePath;

		// Ask user for TRT plan options
		string precision;
		cout << "Choose the precision for the TRT plan (fp16/int8): ";
		cin >> precision;

		string trtPlanPath;
		cout << "Enter the path to save the TRT plan: ";
		cin >> trtPlanPath;

		DynamicShapeRange trtDynamicShapeRange;
		if (dynamic) {
			trtDynamicShapeRange.minDims = nvinfer1::Dims4(minShape[0], minShape[1], minShape[2], minShape[3]);
			trtDynamicShapeRange.optDims = nvinfer1::Dims4(optShape[0], optShape[1], optShape[2], optShape[3]);
			trtDynamicShapeRange.maxDims = nvinfer1::Dims4(maxShape[0], maxShape[1], maxShape[2], maxShape[3]);
		}

		// Generate TRT plan from ONNX model
		TRTGeneration::gen_trt_plan(onnxFilePath, trtPlanPath, precision, dynamic, trtDynamicShapeRange);

		break;
	}
	case 2: {
		// PyTorch to ONNX conversion
		string pretrainedParamsPath;
		cout << "Do you want to enter a path for pretrained parameters? (yes/no): ";
		cin >> userInput;
		if (userInput == "yes" || userInput == "y") {
			cout << "Enter the path to the pretrained parameters: ";
			cin >> pretrainedParamsPath;
		}
		else {
			pretrainedParamsPath = "None";
		}

		vector<int> inputShape;
		cout << "Enter the desired input shape (format: N C H W): ";
		int dim;
		for (int i = 0; i < 4; ++i) {
			cin >> dim;
			inputShape.push_back(dim);
		}

		bool dynamic;
		cout << "Do you intend to have dynamic shapes in this model? (yes/no): ";
		cin >> userInput;
		dynamic = (userInput == "yes" || userInput == "y");

		unordered_map<string, vector<int>> dynamicShapeRange;
		vector<int> minShape, optShape, maxShape;

		if (dynamic) {
			cout << "Enter the minimum shape (format: N C H W): ";
			for (int i = 0; i < 4; ++i) {
				cin >> dim;
				minShape.push_back(dim);
			}
			cout << "Enter the optimum shape (format: N C H W): ";
			for (int i = 0; i < 4; ++i) {
				cin >> dim;
				optShape.push_back(dim);
			}
			cout << "Enter the maximum shape (format: N C H W): ";
			for (int i = 0; i < 4; ++i) {
				cin >> dim;
				maxShape.push_back(dim);
			}
			dynamicShapeRange["min"] = minShape;
			dynamicShapeRange["opt"] = optShape;
			dynamicShapeRange["max"] = maxShape;
		}

		string onnxFilePath;
		cout << "Enter the path to save the ONNX model: ";
		cin >> onnxFilePath;

		break;
	}
	case 3: {
		// ONNX to TRT conversion
		//string onnxFilePath = "../../ONNX-models/mobileone_s4.onnx";
		string onnxFilePath = "D:/csi4900/onnx-models/mobileone_s4.onnx";									// Define your own path
		string trtPlanPath = "D:/csi4900/onnx2trtOld/onnx-to-trt-old/TRT-Plans/mobileone_s4.edhe.plan";		// Define your own path
		//cout << "Enter the path to the ONNX model: ";
		//cin >> onnxFilePath;
		//cout << "Enter the path to save the TRT plan: ";
		//cin >> trtPlanPath;

		// verifying output file path
		std::filesystem::path planDir("D:/csi4900/onnx2trtOld/onnx-to-trt-old/TRT-Plans/");					// Define your own path
		if (std::filesystem::exists(planDir) && std::filesystem::is_directory(planDir)) {
			std::cout << "The Output Path is Good" << std::endl;
		}
		else {
			std::cerr << "The Output Path is invalid" << std::endl;
			return EXIT_FAILURE; // or return 1; or std::exit(EXIT_FAILURE);
		}

		// Ask user for TRT plan options
		string precision;
		cout << "Choose the precision for the TRT plan (fp16/int8): ";
		cin >> precision;

		bool useDynamicShape;
		cout << "Do you want to use dynamic shapes for the TRT plan? (yes/no): ";
		cin >> userInput;
		useDynamicShape = (userInput == "yes" || userInput == "y");

		DynamicShapeRange dynamicShapeRange;
		if (useDynamicShape) {

			cout << "NOTE: If you are using INT8 precision, the calibrator will use the batchSize from optDims" << endl;

			cout << "Enter minDims (format: N C H W): ";
			cin >> dynamicShapeRange.minDims.d[0] >> dynamicShapeRange.minDims.d[1]
				>> dynamicShapeRange.minDims.d[2] >> dynamicShapeRange.minDims.d[3];

			cout << "Enter optDims (format: N C H W): ";
			cin >> dynamicShapeRange.optDims.d[0] >> dynamicShapeRange.optDims.d[1]
				>> dynamicShapeRange.optDims.d[2] >> dynamicShapeRange.optDims.d[3];

			cout << "Enter maxDims (format: N C H W): ";
			cin >> dynamicShapeRange.maxDims.d[0] >> dynamicShapeRange.maxDims.d[1]
				>> dynamicShapeRange.maxDims.d[2] >> dynamicShapeRange.maxDims.d[3];
		}

		// Generate TRT plan from ONNX model
		TRTGeneration::gen_trt_plan(onnxFilePath, trtPlanPath, precision, useDynamicShape, dynamicShapeRange);
		break;
	}
	case 4: {
		// Only run inference
		cout << "Do you want to input a single image or an image directory? (single/directory): ";
		cin >> userInput;

		if (userInput == "single") {

			cout << "Enter the full path of the input image: ";
			cin >> userInput;

			torch::Tensor img_tensor = ImageProcessingUtil::process_img(userInput, false);
			cout << "Input image tensor sizes: " << img_tensor.sizes() << endl;

			cout << "Please enter the full TRT Path that you wish to run inference on: ";
			cin >> planFilePath;

			TRTInference::measure_segmentation_trt_performance_mul(planFilePath, img_tensor, 20);
		}
		else if (userInput == "d") { //"directory") {

			//cout << "Enter the full path of the image directory: ";
			//cin >> userInput;
			char* dir_path = "C:/data_svn/ai_general/code_resource/Hawk/onnx";
			for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {//userInput)) {
				if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
					img_paths.push_back(entry.path().string());
				}
			}

			torch::Tensor img_tensor_batch = ImageProcessingUtil::process_img_batch(img_paths, false);
			cout << "Input image tensor sizes: " << img_tensor_batch.sizes() << endl;

			//cout << "Please enter the full TRT Path that you wish to run inference on: ";
			//cin >> planFilePath;

			TRTInference::measure_segmentation_trt_performance_mul(planFilePath, img_tensor_batch, 20);
		}
		else {
			cout << "Invalid input. Terminating the program." << endl;
			return 0;
		}

		break;
	}
	default:
		cout << "Invalid option. Terminating the program." << endl;
		return 0;
	}

	if (lazyLoading != nullptr) {
		cout << "CUDA_MODULE_LOADING is set to: " << lazyLoading << endl;
	}
	else {
		cout << "CUDA_MODULE_LOADING is not set." << endl;
	}

	return EXIT_SUCCESS;
}