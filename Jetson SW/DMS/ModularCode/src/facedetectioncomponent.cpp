#include "facedetectioncomponent.h"


//constructor
FaceDetectionComponent::FaceDetectionComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


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
    return true;
}

// start detection loop in another thread
void FaceDetectionComponent::startDetection() {
    if (running) {
        std::cerr << "Detection is already running." << std::endl;
        return;
    }
    running = true;
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

    while (running) {
        if (inputQueue.tryPop(frame)) {
            auto start = std::chrono::high_resolution_clock::now();
            detectFaces(frame);
            auto end = std::chrono::high_resolution_clock::now();
    	    double yoloTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "YOLO time:" << std::endl;
    	    std::cout << yoloTime << std::endl;
            //double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            //updatePerformanceMetrics(detectionTime);
            //displayPerformanceMetrics(frame);

            //outputQueue.push(frame);
        }
    }
}


// Function to start the YOLO face detection and crop the first detected face
void FaceDetectionComponent::detectFaces(cv::Mat& frame) {
    cv::Mat blob;
    try {
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        float confThreshold = static_cast<float>(this->fdt) / 100.0f;
        bool firstFaceCropped = false; // Flag to check if the first face is cropped

        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                const float* detection = out.ptr<float>(i);
                float confidence = detection[4];
                if (confidence > confThreshold && !firstFaceCropped) {
                    // Crop the first face and push to the queue, then break
                    cv::Rect faceRect = getFaceRect(detection, frame);
                    cv::Mat faceCrop = frame(faceRect);

                    outputQueue.push(faceCrop); // Assuming outputQueue is thread-safe
                    firstFaceCropped = true;
                    break; // Stop after the first face
                }
            }
            if (firstFaceCropped) break; // Break outer loop if face is already processed
        }
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
    totalFramesProcessed++;

    auto currentTime = std::chrono::high_resolution_clock::now();
    double timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();

    if (timeSinceLastUpdate >= 1000.0) { // Update FPS every second
        fps = totalFramesProcessed / (timeSinceLastUpdate / 1000.0);
        totalFramesProcessed = 0;
        totalDetectionTime = 0;
        lastTime = currentTime;
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


