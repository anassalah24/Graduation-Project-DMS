#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>
#include <numeric> // For std::accumulate
#include <vector>




class HeadPoseComponent {
    // Separate timing and statistics for each engine
    std::vector<double> headPoseTimes, eyeGazeTimes;
    double maxHeadPoseTime = 0.0, maxEyeGazeTime = 0.0;
    double minHeadPoseTime = std::numeric_limits<double>::max(), minEyeGazeTime = std::numeric_limits<double>::max();
    double totalHeadPoseTime = 0.0, totalEyeGazeTime = 0.0;
    size_t headPoseCount = 0, eyeGazeCount = 0;
public:
    HeadPoseComponent(ThreadSafeQueue<cv::Mat>& inputQueue,ThreadSafeQueue<cv::Rect>& faceRectQueue, ThreadSafeQueue<std::vector<std::vector<float>>>& outputQueue,ThreadSafeQueue<cv::Mat>& framesQueue, ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~HeadPoseComponent();

    bool initialize();
    void startHeadPoseDetection();
    void stopHeadPoseDetection();
    void updateHeadPoseEngine(const std::string& headPoseEnginePath);
    void updateEyeGazeEngine(const std::string& eyeGazeEnginePath);
    void logPerformanceMetrics();



private:
    ThreadSafeQueue<cv::Mat>& inputQueue;
    ThreadSafeQueue<cv::Rect>& faceRectQueue;
    ThreadSafeQueue<std::vector<std::vector<float>>>& outputQueue;
    ThreadSafeQueue<cv::Mat>& framesQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;
    std::thread HeadPoseDetectionThread;
    double avgDetectionTime;  // Average time for detection per frame

    bool running;
    
    void HeadPoseDetectionLoop();
    std::vector<std::vector<float>> detectHeadPose(cv::Mat& frame);
    cv::Rect detectFaceRectangle(const cv::Mat& frame);


    
    // members for performance metrics
    double totalDetectionTime = 0;
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;
    void updatePerformanceMetrics(double detectionTime);
    void displayPerformanceMetrics(cv::Mat& frame);
};

