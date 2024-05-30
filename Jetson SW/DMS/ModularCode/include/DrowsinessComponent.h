#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>

class DrowsinessComponent {
public:
    DrowsinessComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~DrowsinessComponent();

    bool initialize();
    void startDrowsinessDetection();
    void stopDrowsinessDetection();
    cv::CascadeClassifier face_cascade;
    cv::Ptr<cv::face::Facemark> facemark;
    // Initialize constants
    const int LEFT_EYE_POINTS[6] = {36, 37, 38, 39, 40, 41};
    const int RIGHT_EYE_POINTS[6] = {42, 43, 44, 45, 46, 47};
    const int MOUTH_EDGE_POINTS[6] = {48, 50, 52, 54, 56, 58};



private:
    ThreadSafeQueue<cv::Mat>& inputQueue;
    ThreadSafeQueue<cv::Mat>& outputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;
    cv::dnn::Net net;
    std::thread drowsinessDetectionThread;

    bool running;
    float fdt=80;
    void drowsinessDetectionLoop();
    void detectDrowsiness(cv::Mat& frame);
    float aspectRatio(const std::vector<cv::Point2f>& landmarks, const int points[]);
    bool isDriverDrowsy(const cv::Mat& faceFrame);
    // members for performance metrics
    double totalDetectionTime = 0;
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;
    void updatePerformanceMetrics(double detectionTime);
    void displayPerformanceMetrics(cv::Mat& frame);
};

