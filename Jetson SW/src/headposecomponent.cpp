#include "headposecomponent.h"
#include "inferseq.h"
#include <boost/filesystem.hpp> // For directory handling with Boost
#include <boost/date_time/posix_time/posix_time.hpp> // For timestamps
#include <boost/date_time/gregorian/gregorian.hpp>  // For to_iso_extended_string for dates
#include <iomanip>  // For std::setw and std::setfill

namespace fs = boost::filesystem;
namespace pt = boost::posix_time;
namespace gr = boost::gregorian;



TRTEngineSingleton* TRTEngineSingleton::instance = nullptr;



//constructor
HeadPoseComponent::HeadPoseComponent(ThreadSafeQueue<cv::Mat>& inputQueue,ThreadSafeQueue<cv::Rect>& faceRectQueue, ThreadSafeQueue<std::vector<std::vector<float>>>& outputQueue,ThreadSafeQueue<cv::Mat>& framesQueue, ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), faceRectQueue(faceRectQueue), outputQueue(outputQueue),framesQueue(framesQueue) ,commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
HeadPoseComponent::~HeadPoseComponent() {
    stopHeadPoseDetection();
}

// initialize model 
bool HeadPoseComponent::initialize() {
    return true;
}

// start detection loop in another thread
void HeadPoseComponent::startHeadPoseDetection() {
    if (running) {
        std::cerr << "headpose Detection is already running." << std::endl;
        return;
    }
    running = true;
    HeadPoseDetectionThread = std::thread(&HeadPoseComponent::HeadPoseDetectionLoop, this);
}

// release thread and any needed cleanup
void HeadPoseComponent::stopHeadPoseDetection() {
    running = false;
    if (HeadPoseDetectionThread.joinable()) {
        HeadPoseDetectionThread.join();
    }
}


void HeadPoseComponent::HeadPoseDetectionLoop() {
    cv::Mat frame;
    bool isFirstFrame = true; // Flag to check if it's the first frame

    while (running) {
        if (inputQueue.tryPop(frame)) {
            cv::Mat croppedFace;
            // Assuming the face is detected and bounded by a rectangle
            cv::Rect faceRect = detectFaceRectangle(frame); // Implement this function to find the rectangle
            if (faceRect.x >= 0 && faceRect.y >= 0 &&
                faceRect.width > 0 && faceRect.height > 0 &&
                faceRect.x + faceRect.width <= frame.cols &&
                faceRect.y + faceRect.height <= frame.rows) {
                croppedFace = frame(faceRect);
            } else {
                croppedFace = frame; // No face detected, use the original frame
            }
            
            auto readings = detectHeadPose(croppedFace);
            framesQueue.push(frame);
            outputQueue.push(readings);

            if (isFirstFrame) {
                inputQueue.clear(); // Clear the queue after the first frame is processed
                isFirstFrame = false; // Update the flag so the queue won't be cleared again
            }
        }
    }
}



std::vector<std::vector<float>> HeadPoseComponent::detectHeadPose(cv::Mat& croppedFace) {
    TRTEngineSingleton* trt = TRTEngineSingleton::getInstance();

    // Measure headpose engine inference time
    auto startHeadPose = std::chrono::high_resolution_clock::now();
    auto headPoseResult = trt->inferHeadPose(croppedFace);
    auto endHeadPose = std::chrono::high_resolution_clock::now();
    double headPoseTime = std::chrono::duration_cast<std::chrono::milliseconds>(endHeadPose - startHeadPose).count();
    headPoseTimes.push_back(headPoseTime);
    maxHeadPoseTime = std::max(maxHeadPoseTime, headPoseTime);
    minHeadPoseTime = std::min(minHeadPoseTime, headPoseTime);
    totalHeadPoseTime += headPoseTime;
    headPoseCount++;

    // Measure eyegaze engine inference time
    auto startEyeGaze = std::chrono::high_resolution_clock::now();
    auto eyeGazeResult = trt->inferEyeGaze(croppedFace);
    auto endEyeGaze = std::chrono::high_resolution_clock::now();
    double eyeGazeTime = std::chrono::duration_cast<std::chrono::milliseconds>(endEyeGaze - startEyeGaze).count();
    eyeGazeTimes.push_back(eyeGazeTime);
    maxEyeGazeTime = std::max(maxEyeGazeTime, eyeGazeTime);
    minEyeGazeTime = std::min(minEyeGazeTime, eyeGazeTime);
    totalEyeGazeTime += eyeGazeTime;
    eyeGazeCount++;

    std::vector<std::vector<float>> out{headPoseResult, eyeGazeResult};
    return out;
}




cv::Rect HeadPoseComponent::detectFaceRectangle(const cv::Mat& frame) {
    cv::Rect faceRect;
    if (!faceRectQueue.tryPop(faceRect)) {
        std::cerr << "No face rectangle found in the queue." << std::endl;
        return cv::Rect(); // Return an empty rectangle
    }
    return faceRect;
}




void HeadPoseComponent::updatePerformanceMetrics(double detectionTime) {
    totalDetectionTime += detectionTime;
    totalFramesProcessed++;

    auto currentTime = std::chrono::high_resolution_clock::now();
    double timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();

    if (timeSinceLastUpdate >= 100.0) { // Update FPS every 100ms
        fps = totalFramesProcessed / (timeSinceLastUpdate / 1000.0);
        totalFramesProcessed = 0;
        totalDetectionTime = 0;
        lastTime = currentTime;
    }
}



void HeadPoseComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time: " + std::to_string(totalDetectionTime / totalFramesProcessed) + " ms";

    cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    //cv::putText(frame, avgTimeText, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    //std::cout << fpsText << std::endl;
    //std::cout << avgTimeText << std::endl;
}





// Update the engine for head pose detection
void HeadPoseComponent::updateHeadPoseEngine(const std::string& headPoseEnginePath) {
    TRTEngineSingleton* trt = TRTEngineSingleton::getInstance();
    trt->setEngine1(headPoseEnginePath);
    std::cout << "Head pose engine updated successfully." << std::endl;
}

// Update the engine for eye gaze detection
void HeadPoseComponent::updateEyeGazeEngine(const std::string& eyeGazeEnginePath) {
    TRTEngineSingleton* trt = TRTEngineSingleton::getInstance();
    trt->setEngine2(eyeGazeEnginePath);
    std::cout << "Eye gaze engine updated successfully." << std::endl;
}


void HeadPoseComponent::logPerformanceMetrics() {


    // Ensure the directory exists
    fs::path dir("benchmarklogs");
    if (!fs::exists(dir)) {
        fs::create_directory(dir);
    }

    // Get current time and format the filename
    pt::ptime now = pt::second_clock::local_time();
    std::ostringstream filename;
    filename << dir.string() << "/benchmark_log_"
             << gr::to_iso_extended_string(now.date()) << "_"  // Correctly use date to string conversion
             << std::setw(2) << std::setfill('0') << now.time_of_day().hours() << "-"
             << std::setw(2) << std::setfill('0') << now.time_of_day().minutes()
             << ".txt";

    // Open the log file in append mode
    std::ofstream logFile(filename.str(), std::ios::app);

    double averageHeadPoseTime = headPoseCount > 0 ? totalHeadPoseTime / headPoseCount : 0;
    double averageEyeGazeTime = eyeGazeCount > 0 ? totalEyeGazeTime / eyeGazeCount : 0;

    logFile << "Head Pose Engine Metrics:\n";
    logFile << "Max Time: " << maxHeadPoseTime << " ms\n";
    logFile << "Min Time: " << minHeadPoseTime << " ms\n";
    logFile << "Average Time: " << averageHeadPoseTime << " ms\n\n";

    logFile << "Eye Gaze Engine Metrics:\n";
    logFile << "Max Time: " << maxEyeGazeTime << " ms\n";
    logFile << "Min Time: " << minEyeGazeTime << " ms\n";
    logFile << "Average Time: " << averageEyeGazeTime << " ms\n\n";

    TRTEngineSingleton* engine = TRTEngineSingleton::getInstance();
    logFile << "Peak GPU Memory Usage for Head Pose: "
            << static_cast<double>(engine->getPeakHeadPoseGpuMemoryUsage()) / (1024 * 1024) << " MB\n";
    logFile << "Peak GPU Memory Usage for Eye Gaze: "
            << static_cast<double>(engine->getPeakEyeGazeGpuMemoryUsage()) / (1024 * 1024) << " MB\n";

    logFile << "Average CPU Memory Usage for Head Pose: "
            << static_cast<double>(engine->gettotalCpuMemoryUsageHeadPose()) / engine->getheadPoseInferenceCount() / (1024 * 1024) << " MB\n";
    logFile << "Average CPU Memory Usage for Eye Gaze: "
            << static_cast<double>(engine->gettotalCpuMemoryUsageEyeGaze()) / engine->geteyeGazeInferenceCount() / (1024 * 1024) << " MB\n";

    logFile << "Average CPU Usage for Head Pose: "
            << static_cast<double>(engine->getheadPoseCpuUsage()) / engine->getheadPoseInferenceCount() << " %\n";

    logFile << "Average CPU Usage for Eye Gaze: "
            << static_cast<double>(engine->geteyeGazeCpuUsage()) / engine->geteyeGazeInferenceCount() << " %\n";

    engine->resetPeakGpuMemoryUsage();

    logFile.close();
}





