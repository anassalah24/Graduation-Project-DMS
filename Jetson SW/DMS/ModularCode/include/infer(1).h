#ifndef INFER_H
#define INFER_H

#include <iostream>
#include <fstream> 
#include <vector> 
#include <NvInfer.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>


using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
} gLogger;

class TRTEngineSingleton {
private:
    static TRTEngineSingleton* instance;
    ICudaEngine* engine;
    IExecutionContext* context; // Member variable for execution context
    bool contextCreated;
    TRTEngineSingleton() : engine(nullptr) {
        // Load TensorRT engine only once
        engine = loadEngine("runovh.engine");
        if (!engine) {
            std::cerr << "Failed to load engine" << std::endl;
            // Handle error appropriately, maybe throw an exception
        }
    }
    cv::cuda::GpuMat preprocessImage(const cv::Mat& inputImage, const std::vector<float>& mean, const std::vector<float>& std, const cv::Size& targetSize) {
    // Upload input image to GPU memory
    cv::cuda::GpuMat gpuInputImage;
    gpuInputImage.upload(inputImage);

    // Resize image
    cv::cuda::GpuMat resizedImage;
    cv::cuda::resize(gpuInputImage, resizedImage, targetSize);

    // Convert to float and normalize
    cv::cuda::GpuMat gpuNormalizedImage;
    resizedImage.convertTo(gpuNormalizedImage, CV_32F, 1.0 / 255.0);
    cv::cuda::GpuMat rgbImage;
    cv::cuda::cvtColor(gpuNormalizedImage, rgbImage, cv::COLOR_BGR2RGB);
    // Compute mean and std values on GPU
    cv::cuda::GpuMat gpuMean(mean.size(), 1, CV_32FC1, const_cast<float*>(mean.data()));
    cv::cuda::GpuMat gpuStd(std.size(), 1, CV_32FC1, const_cast<float*>(std.data()));

    // Normalize image
    cv::cuda::subtract(rgbImage, gpuMean, gpuNormalizedImage);
    cv::cuda::divide(rgbImage, gpuStd, gpuNormalizedImage);
    return rgbImage;
}
    ICudaEngine* loadEngine(const std::string& engineFile) {
        std::ifstream file("/home/dms/DMS/ModularCode/include/Az.engine", std::ios::binary);
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
    void createContext() {
        // Create execution context
        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
        }
    }

public:
    static TRTEngineSingleton* getInstance() {
        if (!instance) {
            instance = new TRTEngineSingleton();
        }
        return instance;
    }

    std::vector<float> infer(cv::Mat& image ) {
        // Create execution context
        if (!contextCreated) {
            createContext();
            contextCreated = true;
        }


        // Read image using OpenCV
        //cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to read image" << std::endl;
            return {-1};
        }

        // Resize image
        cv::Size targetSize(224, 224);

        // Mean and standard deviation values
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};
        cv::cuda::GpuMat gpuNormalizedImage = preprocessImage(image, mean, std, targetSize);

        // Allocate GPU buffers
        int batchSize = 1;
        int inputSize = 1 * 3 * 224 * 224; // Input size for the model
        int outputSize = 9; // Output size for the model

        void* gpuInputBuffer;
        void* gpuOutputBuffer;
        cudaMalloc(&gpuInputBuffer, batchSize * inputSize * sizeof(float));
        cudaMalloc(&gpuOutputBuffer, batchSize * outputSize * sizeof(float));

        // Copy preprocessed image data to GPU
        cudaMemcpy(gpuInputBuffer, gpuNormalizedImage.ptr<float>(), inputSize * sizeof(float), cudaMemcpyDeviceToDevice);

        // Do inference
        void* buffers[] = {gpuInputBuffer, gpuOutputBuffer};
        context->executeV2(buffers);

        // Copy output data from GPU
        std::vector<float> outputData(batchSize * outputSize);

        // Copy data from GPU to host using cudaMemcpy
        cudaMemcpy(outputData.data(), gpuOutputBuffer, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(gpuInputBuffer);
        cudaFree(gpuOutputBuffer);
        return outputData;
        // No need to destroy the engine here since it's managed by the singleton
    }

    ~TRTEngineSingleton() {
        if (context) {
            context->destroy();
            context = nullptr;
        }
        if (engine) {
            engine->destroy();
            engine = nullptr;
        }
    }
};
#endif
