#pragma once

#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <atomic>

class BasicCameraComponent {
public:
    // Constructor
    BasicCameraComponent(ThreadSafeQueue<cv::Mat>& outputQueue, 
                         ThreadSafeQueue<std::string>& commandsQueue, 
                         ThreadSafeQueue<std::string>& faultsQueue);
    
    // Destructor
    ~BasicCameraComponent();

    // Initialize the camera component with the given source
    bool initialize(const std::string& source);
    
    // Start capturing video frames
    void startCapture();
    
    // Stop capturing video frames
    void stopCapture();
    
    // Set the frames per second for capturing
    void setFPS(int fps);
    
    // Set the source for capturing video
    void setSource(const std::string& source);

private:
    cv::VideoCapture cap; // OpenCV video capture object
    std::thread captureThread; // Thread for capturing video
    bool running; // Flag to indicate if capturing is running
    int fps; // Frames per second for capturing
    ThreadSafeQueue<cv::Mat>& outputQueue; // Queue for output frames
    ThreadSafeQueue<std::string>& commandsQueue; // Queue for commands
    ThreadSafeQueue<std::string>& faultsQueue; // Queue for faults

    // Main loop for capturing video frames
    void captureLoop();
};

