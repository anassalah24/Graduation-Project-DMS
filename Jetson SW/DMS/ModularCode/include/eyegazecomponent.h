#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>


class EyeGazeComponent {
public:
    EyeGazeComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputFramesQueue, ThreadSafeQueue<std::string>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~EyeGazeComponent();

    bool initialize();
    void startEyeGazeDetection();
    void stopEyeGazeDetection();



private:
    ThreadSafeQueue<cv::Mat>& inputQueue;
    ThreadSafeQueue<cv::Mat>& outputFramesQueue;
    ThreadSafeQueue<std::string>& outputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;
    std::thread EyeGazeDetectionThread;
    double avgDetectionTime;  // Average time for detection per frame

    bool running;
    
    void EyeGazeDetectionLoop();
    void detectEyeGaze(cv::Mat& frame);

    
    // members for performance metrics
    double totalDetectionTime = 0;
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;
    void updatePerformanceMetrics(double detectionTime);
    void displayPerformanceMetrics(cv::Mat& frame);
};

