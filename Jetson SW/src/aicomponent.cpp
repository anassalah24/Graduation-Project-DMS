#include "aicomponent.h"
#include "infer.h"
#include <boost/filesystem.hpp> 
#include <boost/date_time/posix_time/posix_time.hpp> 
#include <boost/date_time/gregorian/gregorian.hpp>  
#include <iomanip>  

namespace fs = boost::filesystem;
namespace pt = boost::posix_time;
namespace gr = boost::gregorian;

TRTEngineSingleton* TRTEngineSingleton::instance = nullptr;

// Constructor
AIComponent::AIComponent(ThreadSafeQueue<cv::Mat>& inputQueue,
                                     ThreadSafeQueue<cv::Rect>& faceRectQueue,
                                     ThreadSafeQueue<std::vector<std::vector<float>>>& outputQueue,
                                     ThreadSafeQueue<cv::Mat>& framesQueue,
                                     ThreadSafeQueue<std::string>& commandsQueue,
                                     ThreadSafeQueue<std::string>& faultsQueue)
    : inputQueue(inputQueue), faceRectQueue(faceRectQueue), outputQueue(outputQueue),
      framesQueue(framesQueue), commandsQueue(commandsQueue), faultsQueue(faultsQueue), running(false){}

// Destructor
AIComponent::~AIComponent() {
    stopAIDetection();
}



// Start detection loop in another thread
void AIComponent::startAIDetection() {
    if (running) {
        std::cerr << "Head pose detection is already running." << std::endl;
        return;
    }
    running = true;
    AIDetectionThread = std::thread(&AIComponent::AIDetectionLoop, this);
}

// Release thread and any needed cleanup
void AIComponent::stopAIDetection() {
    running = false;
    if (AIDetectionThread.joinable()) {
        AIDetectionThread.join();
    }
}

// Detection loop
void AIComponent::AIDetectionLoop() {
    cv::Mat frame;
    bool isFirstFrame = true; 

    while (running) {
        if (inputQueue.tryPop(frame)) {
            cv::Mat croppedFace;
            cv::Rect faceRect = detectFaceRectangle(frame);
            if (faceRect.x >= 0 && faceRect.y >= 0 &&
                faceRect.width > 0 && faceRect.height > 0 &&
                faceRect.x + faceRect.width <= frame.cols &&
                faceRect.y + faceRect.height <= frame.rows) {
                croppedFace = frame(faceRect);
            } else {
                croppedFace = frame; 
            }
            
            auto readings = detectAI(croppedFace);
            framesQueue.push(frame);
            outputQueue.push(readings);

            if (isFirstFrame) {
                inputQueue.clear(); 
                isFirstFrame = false;
            }
        }
    }
}

// Detect head pose and eye gaze
std::vector<std::vector<float>> AIComponent::detectAI(cv::Mat& croppedFace) {
    TRTEngineSingleton* trt = TRTEngineSingleton::getInstance();

    auto startHeadPose = std::chrono::high_resolution_clock::now();
    auto headPoseResult = trt->inferHeadPose(croppedFace);
    auto endHeadPose = std::chrono::high_resolution_clock::now();
    double headPoseTime = std::chrono::duration_cast<std::chrono::milliseconds>(endHeadPose - startHeadPose).count();
    if (headPoseTime > 45) {
        headPoseTimes.push_back(headPoseTime);
        maxHeadPoseTime = std::max(maxHeadPoseTime, headPoseTime);
        minHeadPoseTime = std::min(minHeadPoseTime, headPoseTime);
        totalHeadPoseTime += headPoseTime;
        headPoseCount++;
    }
    

    // Crop the upper 55% of the image
    int newHeight = static_cast<int>(croppedFace.rows * 0.55);
    cv::Rect roi(0, 0, croppedFace.cols, newHeight);
    cv::Mat upperCroppedFace = croppedFace(roi);
   
    auto startEyeGaze = std::chrono::high_resolution_clock::now();
    auto eyeGazeResult = trt->inferEyeGaze(upperCroppedFace);
    auto endEyeGaze = std::chrono::high_resolution_clock::now();
    double eyeGazeTime = std::chrono::duration_cast<std::chrono::milliseconds>(endEyeGaze - startEyeGaze).count();
    if (eyeGazeTime > 30) {
        eyeGazeTimes.push_back(eyeGazeTime);
        maxEyeGazeTime = std::max(maxEyeGazeTime, eyeGazeTime);
        minEyeGazeTime = std::min(minEyeGazeTime, eyeGazeTime);
        totalEyeGazeTime += eyeGazeTime;
        eyeGazeCount++;
    }

    std::vector<std::vector<float>> out{headPoseResult, eyeGazeResult};
    return out;
}

// Detect face rectangle
cv::Rect AIComponent::detectFaceRectangle(const cv::Mat& frame) {
    cv::Rect faceRect;
    if (!faceRectQueue.tryPop(faceRect)) {
        std::cerr << "No face rectangle found in the queue." << std::endl;
        return cv::Rect(); // Return an empty rectangle
    }
    return faceRect;
}

// Update performance metrics
void AIComponent::updatePerformanceMetrics(double detectionTime) {
    totalDetectionTime += detectionTime;
    totalFramesProcessed++;

    auto currentTime = std::chrono::high_resolution_clock::now();
    double timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();

    if (timeSinceLastUpdate >= 100.0) { 
        fps = totalFramesProcessed / (timeSinceLastUpdate / 1000.0);
        totalFramesProcessed = 0;
        totalDetectionTime = 0;
        lastTime = currentTime;
    }
}

// Display performance metrics
void AIComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time: " + std::to_string(totalDetectionTime / totalFramesProcessed) + " ms";

    cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

}

// Update the engine for head pose detection
void AIComponent::updateHeadPoseEngine(const std::string& headPoseEnginePath) {
    TRTEngineSingleton* trt = TRTEngineSingleton::getInstance();
    trt->setEngine1(headPoseEnginePath);
    std::cout << "Head pose engine updated successfully." << std::endl;
    commandsQueue.push("Clear Queue");
}

// Update the engine for eye gaze detection
void AIComponent::updateEyeGazeEngine(const std::string& eyeGazeEnginePath) {
    TRTEngineSingleton* trt = TRTEngineSingleton::getInstance();
    trt->setEngine2(eyeGazeEnginePath);
    std::cout << "Eye gaze engine updated successfully." << std::endl;
    commandsQueue.push("Clear Queue");
}

// Log performance metrics
void AIComponent::logPerformanceMetrics() {
    fs::path dir("benchmarklogs");
    if (!fs::exists(dir)) {
        fs::create_directory(dir);
    }

    pt::ptime now = pt::second_clock::local_time();
    std::ostringstream filename;
    filename << dir.string() << "/benchmark_log_"
             << gr::to_iso_extended_string(now.date()) << "_"
             << std::setw(2) << std::setfill('0') << now.time_of_day().hours() << "-"
             << std::setw(2) << std::setfill('0') << now.time_of_day().minutes()
             << ".txt";

    std::ofstream logFile(filename.str(), std::ios::app);

    double averageHeadPoseTime = headPoseCount > 0 ? totalHeadPoseTime / headPoseCount : 0;
    double averageEyeGazeTime = eyeGazeCount > 0 ? totalEyeGazeTime / eyeGazeCount : 0;
    if (minHeadPoseTime == std::numeric_limits<double>::max()) { minHeadPoseTime = 0; }
    if (minEyeGazeTime == std::numeric_limits<double>::max()) { minEyeGazeTime = 0; }

    logFile << "<<------------------------------------------------------------------->>\n";
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
    logFile << "<<------------------------------------------------------------------->>\n";

    resetPerformanceMetrics();
    engine->resetPeakGpuMemoryUsage();
    logFile.close();
}

// Reset performance metrics
void AIComponent::resetPerformanceMetrics() {
	totalHeadPoseTime=0.0;
	totalEyeGazeTime= 0.0;
	headPoseCount =0;
	eyeGazeCount=0;
	minHeadPoseTime = std::numeric_limits<double>::max();
	minEyeGazeTime = std::numeric_limits<double>::max();
	maxHeadPoseTime = 0.0;
	maxEyeGazeTime = 0.0;
}

