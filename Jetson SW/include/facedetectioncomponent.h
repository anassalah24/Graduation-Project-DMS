#pragma once

#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>
#include <limits>

class FaceDetectionComponent {
public:
    // Constructor
    FaceDetectionComponent(ThreadSafeQueue<cv::Mat>& inputQueue, 
                           ThreadSafeQueue<cv::Mat>& outputQueue, 
                           ThreadSafeQueue<cv::Rect>& faceRectQueue, 
                           ThreadSafeQueue<std::string>& commandsQueue, 
                           ThreadSafeQueue<std::string>& faultsQueue);
    
    // Destructor
    ~FaceDetectionComponent();

    // Initialize the face detection component
    bool initialize(const std::string& modelConfiguration, const std::string& modelWeights);

    // Start the face detection loop
    void startDetection();

    // Stop the face detection loop
    void stopDetection();

    // Set the face detection threshold
    void setFDT(int fdt);

    // Log performance metrics
    void logPerformanceMetrics();

    // Flag to indicate the status of the model
    bool modelstatus = false;

    // Reset performance metrics
    void resetPerformanceMetrics();

private:
    ThreadSafeQueue<cv::Mat>& inputQueue; // Queue for input frames
    ThreadSafeQueue<cv::Mat>& outputQueue; // Queue for output frames
    ThreadSafeQueue<cv::Rect>& faceRectQueue; // Queue for detected face rectangles
    ThreadSafeQueue<std::string>& commandsQueue; // Queue for commands
    ThreadSafeQueue<std::string>& faultsQueue; // Queue for faults
    cv::dnn::Net net; // DNN network for face detection
    std::thread detectionThread; // Thread for face detection

    bool running; // Flag to indicate if detection is running
    float fdt = 90; // Face detection threshold

    // Main loop for face detection
    void detectionLoop();

    // Function to detect faces in a frame
    void detectFaces(cv::Mat& frame);

    // Helper function to get the rectangle of the face from detection data
    cv::Rect getFaceRect(const float* detection, const cv::Mat& frame);

    int frameCounter = 0; // Counter for frames processed
    int skipRate = 3; // Frame skip rate

    // Members for performance metrics
    double totalDetectionTime = 0;
    double maxDetectionTime = 0;
    double minDetectionTime = std::numeric_limits<double>::max();
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;

    // Function to update performance metrics
    void updatePerformanceMetrics(double detectionTime);

    // Function to display performance metrics on the frame
    void displayPerformanceMetrics(cv::Mat& frame);
};

