#ifndef INFER_H
#define INFER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h> // For cudaMemGetInfo
#include <cuda_runtime.h>
#include <unistd.h>           // For sysconf and _SC_PAGESIZE

#include <fstream>
#include <sstream>
#include <unistd.h> // For sleep function



struct CPUUsage {
    long idleTime;
    long totalTime;
};


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

    size_t peakHeadPoseGpuMemoryUsage = 0;
    size_t peakEyeGazeGpuMemoryUsage = 0;

	size_t totalCpuMemoryUsageHeadPose = 0;
	size_t totalCpuMemoryUsageEyeGaze = 0;
	size_t headPoseInferenceCount = 0;
	size_t eyeGazeInferenceCount = 0;
	size_t totalCpuUsageEyeGaze = 0;
	size_t totalCpuUsageHeadPose = 0;






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

  std::vector<float> inferHeadPose(const cv::Mat& croppedFace) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!engine1) {
        std::cerr << "Head pose engine not loaded, using default values." << std::endl;
        return std::vector<float>(9, -100);
    }


    size_t memUsageBefore = getHostMemoryUsage();
    CPUUsage cpuUsageBefore = getCPUUsage();





    // Preprocessing
    cv::Mat resizedImage;
    cv::resize(croppedFace, resizedImage, cv::Size(224, 224));  // Assume input size is 224x224
    resizedImage.convertTo(resizedImage, CV_32F);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    for (int c = 0; c < 3; ++c) {
        resizedImage.forEach<cv::Vec3f>([&mean, &std, c](cv::Vec3f &pixel, const int* position) -> void {
            pixel[c] = (pixel[c] / 255.0 - mean[c]) / std[c];
        });
    }

    // Allocate GPU buffers
    int batchSize = 1;
    int inputSize = 1 * 3 * 224 * 224;
    int outputSize = 9;  // Output size for head pose model
    void* gpuInputBuffer;
    void* gpuOutputBuffer;
    cudaMalloc(&gpuInputBuffer, batchSize * inputSize * sizeof(float));
    cudaMalloc(&gpuOutputBuffer, batchSize * outputSize * sizeof(float));

    size_t freeMemBefore, totalMemBefore, freeMemAfter, totalMemAfter;
    cudaMemGetInfo(&freeMemBefore, &totalMemBefore); // Get free memory before inference

    // Copy data to GPU
    cudaMemcpy(gpuInputBuffer, resizedImage.ptr<float>(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Execute model
    IExecutionContext* context = engine1->createExecutionContext();
    void* buffers[] = {gpuInputBuffer, gpuOutputBuffer};
    context->executeV2(buffers);

    // Retrieve results
    std::vector<float> results(outputSize);
    cudaMemcpy(results.data(), gpuOutputBuffer, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemGetInfo(&freeMemAfter, &totalMemAfter); // Get free memory after inference

	size_t usedMemDuringInference = (freeMemBefore - freeMemAfter);
	if (usedMemDuringInference > peakHeadPoseGpuMemoryUsage) {
		peakHeadPoseGpuMemoryUsage = usedMemDuringInference;
	}

    size_t memUsageAfter = getHostMemoryUsage();


    size_t memUsageDelta = memUsageAfter - memUsageBefore;


    totalCpuMemoryUsageHeadPose += memUsageDelta;
    headPoseInferenceCount++;
 
	CPUUsage cpuUsageAfter = getCPUUsage();



    double cpuUsage = calculateCPUUsage(cpuUsageBefore, cpuUsageAfter);
    totalCpuUsageHeadPose += cpuUsage;








    // Cleanup
    cudaFree(gpuInputBuffer);
    cudaFree(gpuOutputBuffer);
    context->destroy();

    return results;
}

  std::vector<float> inferEyeGaze(const cv::Mat& croppedFace) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!engine2) {
        std::cerr << "Eye gaze engine not loaded, using default values." << std::endl;
        return std::vector<float>(9, -100);
    }

    size_t memUsageBefore = getHostMemoryUsage();

    CPUUsage cpuUsageBefore = getCPUUsage();



    // Preprocessing
    cv::Mat resizedImage;
    cv::resize(croppedFace, resizedImage, cv::Size(224, 224));
    resizedImage.convertTo(resizedImage, CV_32F);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    std::vector<float> meanValues = {0.485, 0.456, 0.406};
    std::vector<float> stdDevs = {0.229, 0.224, 0.225};
    for (int c = 0; c < 3; ++c) {
        resizedImage.forEach<cv::Vec3f>([&meanValues, &stdDevs, c](cv::Vec3f &pixel, const int* position) -> void {
            pixel[c] = (pixel[c] / 255.0 - meanValues[c]) / stdDevs[c];
        });
    }

    // Allocate GPU buffers
    int batchSize = 1;
    int inputSize = 1 * 3 * 224 * 224;
    int outputSize = 9;  // Assuming the output size for the eye gaze model
    void* gpuInputBuffer;
    void* gpuOutputBuffer;
    cudaMalloc(&gpuInputBuffer, batchSize * inputSize * sizeof(float));
    cudaMalloc(&gpuOutputBuffer, batchSize * outputSize * sizeof(float));

    size_t freeMemBefore, totalMemBefore, freeMemAfter, totalMemAfter;
    cudaMemGetInfo(&freeMemBefore, &totalMemBefore); // Get free memory before inference


    // Copy data to GPU
    cudaMemcpy(gpuInputBuffer, resizedImage.ptr<float>(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Execute model
    IExecutionContext* context = engine2->createExecutionContext();
    void* buffers[] = {gpuInputBuffer, gpuOutputBuffer};
    context->executeV2(buffers);

    // Retrieve results
    std::vector<float> results(outputSize);
    cudaMemcpy(results.data(), gpuOutputBuffer, outputSize * sizeof(float), cudaMemcpyDeviceToHost);



    cudaMemGetInfo(&freeMemAfter, &totalMemAfter); // Get free memory after inference

    size_t usedMemDuringInference = (freeMemBefore - freeMemAfter);
    if (usedMemDuringInference > peakEyeGazeGpuMemoryUsage) {
        peakEyeGazeGpuMemoryUsage = usedMemDuringInference;
    }

    size_t memUsageAfter = getHostMemoryUsage();


    size_t memUsageDelta = memUsageAfter - memUsageBefore;


    totalCpuMemoryUsageEyeGaze += memUsageDelta;
    eyeGazeInferenceCount++;


	CPUUsage cpuUsageAfter = getCPUUsage();




    double cpuUsage = calculateCPUUsage(cpuUsageBefore, cpuUsageAfter);




    totalCpuUsageEyeGaze += cpuUsage;







    // Cleanup
    cudaFree(gpuInputBuffer);
    cudaFree(gpuOutputBuffer);
    context->destroy();

    return results;
}

    size_t getPeakHeadPoseGpuMemoryUsage() const {
        return peakHeadPoseGpuMemoryUsage;
    }

    size_t getPeakEyeGazeGpuMemoryUsage() const {
        return peakEyeGazeGpuMemoryUsage;
    }

    size_t gettotalCpuMemoryUsageHeadPose() const {
        return totalCpuMemoryUsageHeadPose;
    }

    size_t gettotalCpuMemoryUsageEyeGaze() const {
        return totalCpuMemoryUsageEyeGaze;
    }
    size_t getheadPoseInferenceCount() const {
        return headPoseInferenceCount;
    }

    size_t geteyeGazeInferenceCount() const {
        return eyeGazeInferenceCount;
    }

    size_t geteyeGazeCpuUsage() const {
        return totalCpuUsageEyeGaze;
    }

    size_t getheadPoseCpuUsage() const {
        return totalCpuUsageHeadPose;
    }


    void resetPeakGpuMemoryUsage() {
        peakHeadPoseGpuMemoryUsage = 0;
        peakEyeGazeGpuMemoryUsage = 0;
		totalCpuMemoryUsageHeadPose = 0;
        totalCpuMemoryUsageEyeGaze = 0;
		headPoseInferenceCount = 0;
		eyeGazeInferenceCount = 0;
        totalCpuUsageEyeGaze = 0;		
        totalCpuUsageHeadPose = 0;	

    }

	size_t getHostMemoryUsage() {
		std::ifstream statm("/proc/self/statm");
		size_t size, resident, share, text, lib, data, dt;
		statm >> size >> resident >> share >> text >> lib >> data >> dt;
		statm.close();
		return resident * sysconf(_SC_PAGESIZE); // Return resident set size in bytes
	}

	size_t getPageFaults() {
		std::ifstream stat("/proc/self/stat");
		std::string dummy;
		size_t page_faults;
		for (int i = 0; i < 10; ++i) stat >> dummy; // Skip first 9 fields
		stat >> page_faults; // Get the 10th field (minor page faults)
		stat.close();
		return page_faults;
	}

	CPUUsage getCPUUsage() {
		std::ifstream file("/proc/stat");
		std::string line;
		std::getline(file, line);
		std::istringstream ss(line);
		std::string cpu;
		long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
		ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
		file.close();

		long idleTime = idle + iowait;
		long totalTime = user + nice + system + idle + iowait + irq + softirq + steal;

		return {idleTime, totalTime};
	}

	double calculateCPUUsage(const CPUUsage& prev, const CPUUsage& curr) {
		long totalDiff = curr.totalTime - prev.totalTime;
		long idleDiff = curr.idleTime - prev.idleTime;
		return 100.0 * (totalDiff - idleDiff) / totalDiff;
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
