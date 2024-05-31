// BasicCameraComponent.h

#pragma once

#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <atomic>

class BasicCameraComponent {
public:
    BasicCameraComponent(ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~BasicCameraComponent();

    bool initialize(const std::string& source);
    void startCapture();
    void stopCapture();
    void setFPS(int fps);
    void setSource(const std::string& source);

private:
    cv::VideoCapture cap;
    std::thread captureThread;
    bool running;
    int fps;
    ThreadSafeQueue<cv::Mat>& outputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;

    void captureLoop();
};

