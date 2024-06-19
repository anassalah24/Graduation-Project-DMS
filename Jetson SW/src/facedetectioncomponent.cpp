#include "facedetectioncomponent.h"
#include <boost/filesystem.hpp> // For directory handling with Boost
#include <boost/date_time/posix_time/posix_time.hpp> // For timestamps
#include <boost/date_time/gregorian/gregorian.hpp>  // For to_iso_extended_string for dates
#include <iomanip>  // For std::setw and std::setfill

namespace fs = boost::filesystem;
namespace pt = boost::posix_time;
namespace gr = boost::gregorian;

//constructor
FaceDetectionComponent::FaceDetectionComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<cv::Rect>& faceRectQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue), faceRectQueue(faceRectQueue), commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
FaceDetectionComponent::~FaceDetectionComponent() {
    stopDetection();
}

// initialize model , choose backend ( CUDA , OPENCV , OPENCL ) 
bool FaceDetectionComponent::initialize(const std::string& modelConfiguration, const std::string& modelWeights) {
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// CUDA , OPENCV
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);// DNN_TARGET_OPENCL , DNN_TARGET_CUDA , DNN_TARGET_CPU 
    if (net.empty()) {
        std::cerr << "Failed to load the model or config file." << std::endl;
        return false;
    }
    std::string command = "Clear Queue";
    commandsQueue.push(command);
    return true;
}


// start detection loop in another thread
void FaceDetectionComponent::startDetection() {
    if (running) {
        std::cerr << "Detection is already running." << std::endl;
        return;
    }
    running = true;
    std::string command = "Clear Queue";
    commandsQueue.push(command);
    detectionThread = std::thread(&FaceDetectionComponent::detectionLoop, this);
}

// release thread and any needed cleanup
void FaceDetectionComponent::stopDetection() {
    running = false;
    if (detectionThread.joinable()) {
        detectionThread.join();
    }
}


// This loop takes frame from input queue , sends it to detect faces and places it into the output queue
void FaceDetectionComponent::detectionLoop() {
    cv::Mat frame;
    this->lastTime = std::chrono::high_resolution_clock::now(); // Initialize the last time
    std::string command = "Clear Queue";
    commandsQueue.push(command);
    while (running) {
        if (inputQueue.tryPop(frame)) {
            auto start = std::chrono::high_resolution_clock::now();
	    if (this->modelstatus==false){
	       outputQueue.push(frame); // Pass the complete frame with the bounding box
	       continue;
	    }
            auto start = std::chrono::high_resolution_clock::now();
            detectFaces(frame);
            auto end = std::chrono::high_resolution_clock::now();
    	    double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "YOLO time:" << std::endl;
    	    std::cout << detectionTime << std::endl;
            //double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            updatePerformanceMetrics(detectionTime);
            //displayPerformanceMetrics(frame);

            //outputQueue.push(frame);
        }
    }
}


// Function to start the YOLO face detection and send the face with max confidence
void FaceDetectionComponent::detectFaces(cv::Mat& frame) {
    cv::Mat blob;
    try {
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
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

        if (maxConf > static_cast<float>(this->fdt) / 100.0f) {
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

	if (detectionTime>0){
		totalFramesProcessed++;
		maxDetectionTime = std::max(maxDetectionTime, detectionTime);
		minDetectionTime = std::min(minDetectionTime, detectionTime);
	}
}

void FaceDetectionComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time: " + std::to_string(totalDetectionTime / totalFramesProcessed) + " ms";

    //cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    //cv::putText(frame, avgTimeText, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    //std::cout << fpsText << std::endl;
    //std::cout << avgTimeText << std::endl;
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
             << gr::to_iso_extended_string(now.date()) << "_"  // Correctly use date to string conversion
             << std::setw(2) << std::setfill('0') << now.time_of_day().hours() << "-"
             << std::setw(2) << std::setfill('0') << now.time_of_day().minutes()
             << ".txt";

    // Open the log file in append mode
    std::ofstream logFile(filename.str(), std::ios::app);

    double averageDetectionTime = totalFramesProcessed > 0 ? totalDetectionTime / totalFramesProcessed : 0;
	
	if (minDetectionTime == std::numeric_limits<double>::max()){minDetectionTime=0;}
    logFile << "Face Detection Metrics:\n";
    logFile << "Max Detection Time: " << maxDetectionTime << " ms\n";
    logFile << "Min Detection Time: " << minDetectionTime << " ms\n";
    logFile << "Average Detection Time: " << averageDetectionTime << " ms\n";
    resetPerformanceMetrics();
    logFile << "<<------------------------------------------------------------------->>\n";
    logFile.close();
}



