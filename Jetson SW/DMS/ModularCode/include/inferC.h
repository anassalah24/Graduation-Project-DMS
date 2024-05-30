#ifndef INFERC_H
#define INFERC_H

#include <iostream>
#include <fstream> 
#include <vector> 
#include <NvInfer.h> 
#include <opencv2/opencv.hpp>
#include <string>

using namespace nvinfer1;

const std::vector<std::string> classes_labels = {
	"Eyes Closed",
	"Forward",
	"Left Mirror",
	"Radio",
	"Rearview",
	"Right Mirror",
	"Shoulder",
	"Speedometer"
};

class LoggerC : public ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity != Severity::kINFO) std::cout << msg << std::endl;
	}
} gLoggerC;

class TRTEngineSingletonC {
private:
	static TRTEngineSingletonC* instance;
	ICudaEngine* engine;
	TRTEngineSingletonC() : engine(nullptr) {
		// Load TensorRT engine only once
		engine = loadEngine("mobilenetV3_engine.engine");
		if (!engine) {
			std::cerr << "Failed to load engine" << std::endl;
			// Handle error appropriately, maybe throw an exception
		}
	}

public:
	static TRTEngineSingletonC* getInstance() {
		if (!instance) {
			instance = new TRTEngineSingletonC();
		}
		return instance;
	}

	std::vector<float> infer(const cv::Mat& frame) {
		// Create execution context
		IExecutionContext* context = engine->createExecutionContext();
		if (!context) {
			std::cerr << "Failed to create execution context" << std::endl;
			return { -1 };
		}

		// Read image using OpenCV
		cv::Mat image = frame;
		if (image.empty()) {
			std::cerr << "Failed to read image" << std::endl;
			return { -1 };
		}

		// Resize image
		cv::Mat resizedImage;
		cv::resize(image, resizedImage, cv::Size(224, 224)); // Resize to input size (224x224)

		// Convert to float and normalize
		resizedImage.convertTo(resizedImage, CV_32F);
		cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB); // OpenCV uses BGR by default, convert to RGB

		// Mean and standard deviation values
		std::vector<float> mean = { 0.485, 0.456, 0.406 };
		std::vector<float> std = { 0.229, 0.224, 0.225 };

		// Normalize image
		for (int c = 0; c < 3; ++c) {
			resizedImage.forEach<cv::Vec3f>([&mean, &std, c](cv::Vec3f& pixel, const int* position) -> void {
				pixel[c] = (pixel[c] / 255.0 - mean[c]) / std[c];
				});
		}

		// Allocate GPU buffers
		int batchSize = 1;
		int inputSize = 1 * 3 * 224 * 224; // Input size for the model
		int outputSize = 8; // Output size for the model

		void* gpuInputBuffer;
		void* gpuOutputBuffer;
		cudaMalloc(&gpuInputBuffer, batchSize * inputSize * sizeof(float));
		cudaMalloc(&gpuOutputBuffer, batchSize * outputSize * sizeof(float));

		// Copy preprocessed image data to GPU
		cudaMemcpy(gpuInputBuffer, resizedImage.ptr<float>(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

		// Do inference
		void* buffers[] = { gpuInputBuffer, gpuOutputBuffer };
		context->executeV2(buffers);

		// Copy output data from GPU
		std::vector<float> outputData(batchSize * outputSize);

		// Copy data from GPU to host using cudaMemcpy
		cudaMemcpy(outputData.data(), gpuOutputBuffer, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

		// Find the index with the highest score
		int maxIndex = 0;
		float maxScore = outputData[0];
		for (int i = 1; i < outputSize; ++i) {
			if (outputData[i] > maxScore) {
				maxIndex = i;
				maxScore = outputData[i];
			}
		}

		// Display the output class label
		std::string outputLabel = classes_labels[maxIndex];
		std::cout << "Output class label: " << outputLabel << std::endl;

		cudaFree(gpuInputBuffer);
		cudaFree(gpuOutputBuffer);
		context->destroy();
		return outputData;
		// No need to destroy the engine here since it's managed by the singleton
	}

	~TRTEngineSingletonC() {
		if (engine) {
			engine->destroy();
			engine = nullptr;
		}
	}

private:
	ICudaEngine* loadEngine(const std::string& engineFile) {
        std::ifstream file("/home/dms/DMS/ModularCode/modelconfigs/mobilenetv3_engine.engine", std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Error opening engine file" << std::endl;
			return nullptr;
		}

		std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
		file.close();

		IRuntime* runtime = createInferRuntime(gLoggerC);
		if (!runtime) {
			std::cerr << "Failed to create InferRuntime" << std::endl;
			return nullptr;
		}

		return runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
	}
};

#endif
