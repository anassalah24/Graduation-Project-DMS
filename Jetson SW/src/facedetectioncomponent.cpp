#include "facedetectioncomponent.h"
#include <boost/filesystem.hpp> 
#include <boost/date_time/posix_time/posix_time.hpp> 
#include <boost/date_time/gregorian/gregorian.hpp> 
#include <iomanip> 


namespace fs = boost::filesystem;
namespace pt = boost::posix_time;
namespace gr = boost::gregorian;

// Constructor
FaceDetectionComponent::FaceDetectionComponent(ThreadSafeQueue<cv::Mat>& inputQueue, 
                                               ThreadSafeQueue<cv::Mat>& outputQueue,
                                               ThreadSafeQueue<cv::Rect>& faceRectQueue,
                                               ThreadSafeQueue<std::string>& commandsQueue,
                                               ThreadSafeQueue<std::string>& faultsQueue)
    : inputQueue(inputQueue), outputQueue(outputQueue), faceRectQueue(faceRectQueue), 
      commandsQueue(commandsQueue), faultsQueue(faultsQueue), running(false){}

// Destructor
FaceDetectionComponent::~FaceDetectionComponent() {
    stopDetection();
}


// Initialize model, choose backend (CUDA, OPENCV, OPENCL)
bool FaceDetectionComponent::initialize(const std::string& modelConfiguration, const std::string& modelWeights) {
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    if (net.empty()) {
        std::cerr << "Failed to load the model or config file." << std::endl;
        return false;
    }
    commandsQueue.push("Clear Queue");
    return true;
}


// Start detection loop in another thread
void FaceDetectionComponent::startDetection() {
    if (running) {
        std::cerr << "Detection is already running." << std::endl;
        return;
    }
    running = true;
    commandsQueue.push("Clear Queue");
    detectionThread = std::thread(&FaceDetectionComponent::detectionLoop, this);
}

// Release thread and any needed cleanup
void FaceDetectionComponent::stopDetection() {
    running = false;
    if (detectionThread.joinable()) {
        detectionThread.join();
    }
}

void FaceDetectionComponent::detectionLoop() {
    cv::Mat frame;
    lastTime = std::chrono::high_resolution_clock::now();
    commandsQueue.push("Clear Queue");
    while (running) {
        if (inputQueue.tryPop(frame)) {
            if (!modelstatus) {
                outputQueue.push(frame);
                continue;
            }
            auto start = std::chrono::high_resolution_clock::now();
            detectFaces(frame);
            auto end = std::chrono::high_resolution_clock::now();
            double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            updatePerformanceMetrics(detectionTime);
        }
    }
}

// Function to start the YOLO face detection and send the face with max confidence
void FaceDetectionComponent::detectFaces(cv::Mat& frame) {
    cv::Mat blob;
    try {
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        float maxConf = 0;
        cv::Rect bestFaceRect;
        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                const float* detection = out.ptr<float>(i);
                float confidence = detection[4];
                if (confidence > maxConf) {
                    maxConf = confidence;
                    bestFaceRect = getFaceRect(detection, frame);
                }
            }
        }

        if (maxConf > static_cast<float>(fdt) / 100.0f) {
            cv::rectangle(frame, bestFaceRect, cv::Scalar(0, 255, 0), 2);
            faceRectQueue.push(bestFaceRect); // Push the bounding box coordinates
        }
        outputQueue.push(frame); // Pass the complete frame with the bounding box
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }
}

// Helper function to calculate the rectangle of the face from detection data
cv::Rect FaceDetectionComponent::getFaceRect(const float* detection, const cv::Mat& frame) {
    int centerX = static_cast<int>(detection[0] * frame.cols);
    int centerY = static_cast<int>(detection[1] * frame.rows);
    int width = static_cast<int>(detection[2] * frame.cols);
    int height = static_cast<int>(detection[3] * frame.rows);
    int left = centerX - width / 2;
    int top = centerY - height / 2;

    return cv::Rect(left, top, width, height);
}

void FaceDetectionComponent::updatePerformanceMetrics(double detectionTime) {
    totalDetectionTime += detectionTime;

    if (detectionTime > 0) {
        totalFramesProcessed++;
        maxDetectionTime = std::max(maxDetectionTime, detectionTime);
        minDetectionTime = std::min(minDetectionTime, detectionTime);
    }
}

void FaceDetectionComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time: " + std::to_string(totalDetectionTime / totalFramesProcessed) + " ms";
}

void FaceDetectionComponent::setFDT(int fdt) {
    this->fdt = fdt;
    std::cout << "FDT CHANGED SUCCESSFULLY" << std::endl;
}

void FaceDetectionComponent::logPerformanceMetrics() {
    // Ensure the directory exists
    fs::path dir("benchmarklogs");
    if (!fs::exists(dir)) {
        fs::create_directory(dir);
    }

    // Get current time and format the filename
    pt::ptime now = pt::second_clock::local_time();
    std::ostringstream filename;
    filename << dir.string() << "/benchmark_log_"
             << gr::to_iso_extended_string(now.date()) << "_"
             << std::setw(2) << std::setfill('0') << now.time_of_day().hours() << "-"
             << std::setw(2) << std::setfill('0') << now.time_of_day().minutes()
             << ".txt";

    // Open the log file in append mode
    std::ofstream logFile(filename.str(), std::ios::app);

    double averageDetectionTime = totalFramesProcessed > 0 ? totalDetectionTime / totalFramesProcessed : 0;
    if (minDetectionTime == std::numeric_limits<double>::max()) {
        minDetectionTime = 0;
    }
    logFile << "Face Detection Metrics:\n";
    logFile << "Max Detection Time: " << maxDetectionTime << " ms\n";
    logFile << "Min Detection Time: " << minDetectionTime << " ms\n";
    logFile << "Average Detection Time: " << averageDetectionTime << " ms\n";
    resetPerformanceMetrics();
    logFile << "<<------------------------------------------------------------------->>\n";
    logFile.close();
}

void FaceDetectionComponent::resetPerformanceMetrics() {
    totalDetectionTime = 0.0;
    totalFramesProcessed = 0;
    maxDetectionTime = 0.0;
    minDetectionTime = std::numeric_limits<double>::max();
}



