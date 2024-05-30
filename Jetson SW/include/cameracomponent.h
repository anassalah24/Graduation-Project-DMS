// CameraComponent.h

#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ICameraComponent {
public:
    virtual ~ICameraComponent() = default;

    // Initialize the camera with specified parameters
    virtual bool initialize(const std::string& source) = 0;

    // Start capturing video frames
    virtual bool startCapture() = 0;

    // Stop capturing video frames
    virtual void stopCapture() = 0;

    // Retrieve the next frame from the camera
    virtual cv::Mat getNextFrame() = 0;

    // Check if the camera component is running
    virtual bool isRunning() const = 0;

    virtual void setFrameRate(double fps) = 0;
};

