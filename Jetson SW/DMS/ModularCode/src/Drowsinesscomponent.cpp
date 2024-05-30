#include "DrowsinessComponent.h"




//constructor
DrowsinessComponent::DrowsinessComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
DrowsinessComponent::~DrowsinessComponent() {
    stopDrowsinessDetection();
}

// initialize model 
bool DrowsinessComponent::initialize() {

    std::string face_cascade_name = ("/home/dms/DMS + AI trial/ModularCode/modelconfigs/haarcascade_frontalface_alt.xml" );
    std::string facemark_filename = "/home/dms/DMS + AI trial/ModularCode/modelconfigs/lbfmodel.yaml";

    facemark = cv::face::createFacemarkLBF();
    facemark -> loadModel(facemark_filename);
    std::cout << "Loaded facemark LBF model" << std::endl;

    if( !face_cascade.load( face_cascade_name ) )
    {
        std::cout << "--(!)Error loading face cascade\n";
        return false;
    };
    
    return true;

    
}

// start detection loop in another thread
void DrowsinessComponent::startDrowsinessDetection() {
    if (running) {
        std::cerr << "Detection is already running." << std::endl;
        return;
    }
    running = true;
    drowsinessDetectionThread = std::thread(&DrowsinessComponent::drowsinessDetectionLoop, this);
}

// release thread and any needed cleanup
void DrowsinessComponent::stopDrowsinessDetection() {
    running = false;
    if (drowsinessDetectionThread.joinable()) {
        drowsinessDetectionThread.join();
    }
}


// This loop takes frame from input queue , sends it to detect faces and places it into the output queue
void DrowsinessComponent::drowsinessDetectionLoop() {
    cv::Mat frame;
    this->lastTime = std::chrono::high_resolution_clock::now(); // Initialize the last time

    while (running) {
        if (inputQueue.tryPop(frame)) {



            auto start = std::chrono::high_resolution_clock::now();
            detectDrowsiness(frame);
            auto end = std::chrono::high_resolution_clock::now();

            double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            updatePerformanceMetrics(detectionTime);
            displayPerformanceMetrics(frame);





            outputQueue.push(frame);
        }
    }
}


// Helper function to calculate the aspect ratio of eye or mouth
float DrowsinessComponent::aspectRatio(const std::vector<cv::Point2f>& landmarks, const int points[]) {
    cv::Point left = landmarks[points[0]];
    cv::Point right = landmarks[points[3]];
    cv::Point top = (landmarks[points[1]] + landmarks[points[2]]) * 0.5;
    cv::Point bottom = (landmarks[points[4]] + landmarks[points[5]]) * 0.5;

    float width = cv::norm(cv::Mat(left), cv::Mat(right));
    float height = cv::norm(cv::Mat(top), cv::Mat(bottom));
    return width / height;
}


bool DrowsinessComponent::isDriverDrowsy(const cv::Mat& faceFrame) {
    cv::Mat gray;
    cvtColor(faceFrame, gray, cv::COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    if (face_cascade.empty()) { // Check if face_cascade is loaded correctly
        return false;
    }
    // Detecting the face again is optional, depending on whether the frame is guaranteed to be pre-cropped to just the face.
    face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    if (!faces.empty()) {
        std::vector<std::vector<cv::Point2f>> shapes;
        if (facemark->fit(faceFrame, faces, shapes)) {
            // Check for blinking
            float leftEyeRatio = aspectRatio(shapes[0], LEFT_EYE_POINTS);
            float rightEyeRatio = aspectRatio(shapes[0], RIGHT_EYE_POINTS);
            bool isBlinking = (leftEyeRatio > 3 || rightEyeRatio > 3);  // Threshold may need tuning

            // Check for yawning
            float mouthRatio = aspectRatio(shapes[0], MOUTH_EDGE_POINTS);
            bool isYawning = mouthRatio < 0.9;  // Threshold may need tuning

            return isBlinking || isYawning;
        }
    }
    return false;
}


//function to start the drowsiness detection
void DrowsinessComponent::detectDrowsiness(cv::Mat& frame) {
    

     bool drowsy = isDriverDrowsy(frame);
     std::string alertText = "State: " + std::string(drowsy ? "Drowsy" : "Not Drowsy");
     cv::putText(frame, alertText , cv::Point(20, 10), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 0), 1);
     std::cout << alertText << std::endl;
}






void DrowsinessComponent::updatePerformanceMetrics(double detectionTime) {
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

void DrowsinessComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time: " + std::to_string(totalDetectionTime / totalFramesProcessed) + " ms";

    cv::putText(frame, fpsText, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 0), 1);
    cv::putText(frame, avgTimeText, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 0), 1);
    std::cout << fpsText << std::endl;
    std::cout << avgTimeText << std::endl;
}


