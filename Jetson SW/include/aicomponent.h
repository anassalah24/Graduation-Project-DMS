#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>
#include <numeric>
#include <vector>
#include <limits>

class AIComponent {
public:
    // Constructor
    AIComponent(ThreadSafeQueue<cv::Mat>& inputQueue, 
                      ThreadSafeQueue<cv::Rect>& faceRectQueue, 
                      ThreadSafeQueue<std::vector<std::vector<float>>>& outputQueue, 
                      ThreadSafeQueue<cv::Mat>& framesQueue, 
                      ThreadSafeQueue<std::string>& commandsQueue, 
                      ThreadSafeQueue<std::string>& faultsQueue);

    // Destructor
    ~AIComponent();

    // Initialize the head pose detection component
    bool initialize();

    // Start the head pose detection loop
    void startAIDetection();

    // Stop the head pose detection loop
    void stopAIDetection();

    // Update the head pose engine
    void updateHeadPoseEngine(const std::string& headPoseEnginePath);

    // Update the eye gaze engine
    void updateEyeGazeEngine(const std::string& eyeGazeEnginePath);

    // Log performance metrics
    void logPerformanceMetrics();

    // Reset performance metrics
    void resetPerformanceMetrics();

private:
    ThreadSafeQueue<cv::Mat>& inputQueue; // Queue for input frames
    ThreadSafeQueue<cv::Rect>& faceRectQueue; // Queue for detected face rectangles
    ThreadSafeQueue<std::vector<std::vector<float>>>& outputQueue; // Queue for output data
    ThreadSafeQueue<cv::Mat>& framesQueue; // Queue for frames
    ThreadSafeQueue<std::string>& commandsQueue; // Queue for commands
    ThreadSafeQueue<std::string>& faultsQueue; // Queue for faults
    std::thread AIDetectionThread; // Thread for head pose detection

    bool running; // Flag to indicate if detection is running
    double avgDetectionTime; // Average time for detection per frame

    // Main loop for head pose detection
    void AIDetectionLoop();

    // Function to detect head pose in a frame
    std::vector<std::vector<float>> detectAI(cv::Mat& frame);

    // Helper function to detect the face rectangle in a frame
    cv::Rect detectFaceRectangle(const cv::Mat& frame);

    // Members for performance metrics
    double totalDetectionTime = 0;
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;

    // Function to update performance metrics
    void updatePerformanceMetrics(double detectionTime);

    // Function to display performance metrics on the frame
    void displayPerformanceMetrics(cv::Mat& frame);

    // Separate timing and statistics for each engine
    std::vector<double> headPoseTimes, eyeGazeTimes;
    double maxHeadPoseTime = 0.0, maxEyeGazeTime = 0.0;
    double minHeadPoseTime = std::numeric_limits<double>::max(), minEyeGazeTime = std::numeric_limits<double>::max();
    double totalHeadPoseTime = 0.0, totalEyeGazeTime = 0.0;
    size_t headPoseCount = 0, eyeGazeCount = 0;
};

