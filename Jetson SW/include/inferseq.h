#ifndef INFER_H
#define INFER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
} gLogger;

class TRTEngineSingleton {
private:
    static TRTEngineSingleton* instance;
    static std::mutex mtx; // Mutex for thread safety
    ICudaEngine* engine1; // Head pose engine
    ICudaEngine* engine2; // Eye gaze engine

    TRTEngineSingleton() : engine1(nullptr), engine2(nullptr) {
        loadEngines(); // Initial engine loading
    }

public:
    static TRTEngineSingleton* getInstance() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!instance) {
            instance = new TRTEngineSingleton();
        }
        return instance;
    }

    void loadEngines() {
        engine1 = loadEngine("/home/dms/DMS/ModularCode/include/Ay.engine");
        engine2 = loadEngine("/home/dms/DMS/ModularCode/modelconfigs/mobilenetv3_engine.engine");
		std::cout << "loaded engines successfullyyyyyyyyyyyyyyyyyyyyyyyyyy " << std::endl;
        if (!engine1 || !engine2) {
            std::cerr << "Failed to load one or both engines" << std::endl;
        }
    }

	void setEngine1(const std::string& enginePath) {
		std::lock_guard<std::mutex> lock(mtx);
		if (enginePath == "No Head Pose") {
			std::cout << "Skipping load of head pose engine." << std::endl;
			if (engine1) {
				engine1->destroy();
				engine1 = nullptr;
			}
			return;
		}
		std::cout << "Loading new engine for head pose from: " << enginePath << std::endl;
		if (engine1) {
		engine1->destroy();
		engine1 = nullptr;
		}
		ICudaEngine* newEngine = loadEngine(enginePath);
		if (newEngine) {
		engine1 = newEngine;
		std::cout << "Updated engine for head pose successfully." << std::endl;
		} else {
		std::cerr << "Failed to load new engine for head pose from: " << enginePath << std::endl;
		}
	}

	void setEngine2(const std::string& enginePath) {
		std::lock_guard<std::mutex> lock(mtx);
		if (enginePath == "No Eye Gaze") {
			std::cout << "Skipping load of eye gaze engine." << std::endl;
			if (engine2) {
				engine2->destroy();
				engine2 = nullptr;
			}
			return;
			}
		std::cout << "Loading new engine for eye gaze from: " << enginePath << std::endl;
		if (engine2) {
		engine2->destroy();
		engine2 = nullptr;
		}
		ICudaEngine* newEngine = loadEngine(enginePath);
		if (newEngine) {
		engine2 = newEngine;
		std::cout << "Updated engine for eye gaze successfully." << std::endl;
		} else {
		std::cerr << "Failed to load new engine for eye gaze from: " << enginePath << std::endl;
		}
	}

    std::vector<std::vector<float>> infer(const cv::Mat& frame) {

		std::lock_guard<std::mutex> lock(mtx); // Protect inference process
		std::vector<float> defaultOutput(9, -100); // Default output when engine is skipped
		std::vector<std::vector<float>> results(2, defaultOutput); // Create execution contexts
		// Create execution contexts
		IExecutionContext* context1 = engine1 ? engine1->createExecutionContext() : nullptr;
		IExecutionContext* context2 = engine2 ? engine2->createExecutionContext() : nullptr;

        // Read and preprocess image using OpenCV
        cv::Mat image = frame;
        if (image.empty()) {
            std::cerr << "Failed to read image" << std::endl;
            return {{-1}, {-1}};
        }

        // Resize and preprocess image
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(224, 224)); // Resize to input size (224x224)
        resizedImage.convertTo(resizedImage, CV_32F);
        cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB); // OpenCV uses BGR by default, convert to RGB

        // Mean and standard deviation values
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};

        // Normalize image
        for (int c = 0; c < 3; ++c) {
            resizedImage.forEach<cv::Vec3f>([&mean, &std, c](cv::Vec3f &pixel, const int* position) -> void {
                pixel[c] = (pixel[c] / 255.0 - mean[c]) / std[c];
            });
        }

        // Allocate GPU buffers
        int batchSize = 1;
        int inputSize = 1 * 3 * 224 * 224; // Input size for the model
        int outputSize1 = 9; // Output size for the first model
        int outputSize2 = 9; // Output size for the second model (assuming same output size)

        void* gpuInputBuffer;
        void* gpuOutputBuffer1;
        void* gpuOutputBuffer2;
        cudaMalloc(&gpuInputBuffer, batchSize * inputSize * sizeof(float));
        cudaMalloc(&gpuOutputBuffer1, batchSize * outputSize1 * sizeof(float));
        cudaMalloc(&gpuOutputBuffer2, batchSize * outputSize2 * sizeof(float));

        // Copy preprocessed image data to GPU
        cudaMemcpy(gpuInputBuffer, resizedImage.ptr<float>(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

		// Inference for the first model
		if (context1) {
		    void* buffers1[] = {gpuInputBuffer, gpuOutputBuffer1};
		    context1->executeV2(buffers1);

		    // Copy output data from GPU
		    cudaMemcpy(results[0].data(), gpuOutputBuffer1, outputSize1 * sizeof(float), cudaMemcpyDeviceToHost);
		} else {
		    std::cerr << "Head pose engine not loaded, using default values." << std::endl;
		}

		// Inference for the second model
		if (context2) {
		    void* buffers2[] = {gpuInputBuffer, gpuOutputBuffer2};
		    context2->executeV2(buffers2);

		    // Copy output data from GPU
		    cudaMemcpy(results[1].data(), gpuOutputBuffer2, outputSize2 * sizeof(float), cudaMemcpyDeviceToHost);
		} else {
		    std::cerr << "Eye gaze engine not loaded, using default values." << std::endl;
		}

        cudaFree(gpuInputBuffer);
        cudaFree(gpuOutputBuffer1);
        cudaFree(gpuOutputBuffer2);
    	if (context1) context1->destroy();
    	if (context2) context2->destroy();

        return results;

        // No need to destroy the engines here since they're managed by the singleton
    }


    ~TRTEngineSingleton() {
        std::lock_guard<std::mutex> lock(mtx);
        if (engine1) {
            engine1->destroy();
            engine1 = nullptr;
        }
        if (engine2) {
            engine2->destroy();
            engine2 = nullptr;
        }
    }

private:
    ICudaEngine* loadEngine(const std::string& engineFile) {
        std::ifstream file(engineFile, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening engine file" << std::endl;
            return nullptr;
        }

        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
        file.close();

        IRuntime* runtime = createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "Failed to create InferRuntime" << std::endl;
            return nullptr;
        }

        return runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    }
};

std::mutex TRTEngineSingleton::mtx;
//extern TRTEngineSingleton* TRTEngineSingleton::instance=null;

#endif
