#pragma once

#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>

class FaceDetectionComponent {
public:
    FaceDetectionComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<cv::Rect>& faceRectQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~FaceDetectionComponent();

    bool initialize(const std::string& modelConfiguration, const std::string& modelWeights);
    void startDetection();
    void stopDetection();
    void setFDT(int fdt);


private:
    ThreadSafeQueue<cv::Mat>& inputQueue;
    ThreadSafeQueue<cv::Mat>& outputQueue;
    ThreadSafeQueue<cv::Rect>& faceRectQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;
    cv::dnn::Net net;
    std::thread detectionThread;
    bool running;
    float fdt=90;
    void detectionLoop();
    void detectFaces(cv::Mat& frame);
    cv::Rect getFaceRect(const float* detection, const cv::Mat& frame);
    //void processDetections(const std::vector<cv::Mat>& outs, cv::Mat& frame, float confThreshold);
    //void drawDetection(cv::Mat& frame, const float* detection);
    int frameCounter = 0;
    int skipRate = 3;
    
    // New members for performance metrics
    double totalDetectionTime = 0;
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;
    void updatePerformanceMetrics(double detectionTime);
    void displayPerformanceMetrics(cv::Mat& frame);
};

