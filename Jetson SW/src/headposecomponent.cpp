#include "headposecomponent.h"
#include "inferseq.h"


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
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> out = trt->infer(croppedFace);
    auto end = std::chrono::high_resolution_clock::now();
    double engineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "AI Models combined time: " << engineTime << " ms" << std::endl;
    return out;
}







    //std::string timeText = "time: " + std::to_string(int(engineTime))+"ms";
    //cv::putText(frame, timeText, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    //double fps = 1000/engineTime;
    //std::string fpsText = "FPS: " + std::to_string(int(fps));
    //cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    //int textPositionY = 55;  // Initial Y position for the first text

    //for (float x : out) {
	    //std::string text = std::to_string(x);  // Convert the float to string

	    // Put the text on the image
	    //cv::putText(frame, text, cv::Point(20, textPositionY), 
		        //cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);

	    //textPositionY += 15;  // Move down for the next text
	    //if (textPositionY > frame.rows - 15) {  // Check if the position is out of image bounds
		//break;  // Reset position or break if you don't want to overwrite
	    //}
    //}


    //for (float x : out){
	//printf("%f \n",x);
	//}

    //for (const auto& row : out) {
        // Loop through each element in the sub-vector
        //for (const auto& elem : row) {
          //  std::cout << elem << " ";
        //}
        //std::cout << std::endl;  // New line for each row
   // }



	
   //outputQueue.push(out);




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


