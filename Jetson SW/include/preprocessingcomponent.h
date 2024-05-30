// PreprocessingComponent.h

#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <string>

class IPreprocessingComponent {
public:
    virtual ~IPreprocessingComponent() = default;
    virtual cv::Mat processFrame(const cv::Mat& inputFrame) = 0;
    virtual void setParameter(const std::string& param, double value) = 0;
};

