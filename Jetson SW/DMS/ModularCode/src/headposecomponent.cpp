#include "headposecomponent.h"
#include "infer.h"


TRTEngineSingleton* TRTEngineSingleton::instance = nullptr;

//constructor
HeadPoseComponent::HeadPoseComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<std::string>& outputQueue,ThreadSafeQueue<cv::Mat>& framesQueue, ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),framesQueue(framesQueue) ,commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


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


// This loop takes frame from input queue , sends it to detect faces and places it into the output queue
void HeadPoseComponent::HeadPoseDetectionLoop() {
    cv::Mat frame;
    this->lastTime = std::chrono::high_resolution_clock::now(); // Initialize the last time
    bool isFirstFrame = true; // Flag to check if it's the first frame

    while (running) {
        if (inputQueue.tryPop(frame)) {



            auto start = std::chrono::high_resolution_clock::now();
            detectHeadPose(frame);
            auto end = std::chrono::high_resolution_clock::now();

            double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            //updatePerformanceMetrics(detectionTime);
            //displayPerformanceMetrics(frame);





            framesQueue.push(frame);

          if (isFirstFrame) {
                inputQueue.clear(); // Clear the queue after the first frame is processed
                isFirstFrame = false; // Update the flag so the queue won't be cleared again
            }
        }
    }
}





//function to start the head pose detection
void HeadPoseComponent::detectHeadPose(cv::Mat& frame) {

    TRTEngineSingleton* trt=TRTEngineSingleton::getInstance();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> out = trt->infer(frame);
    auto end = std::chrono::high_resolution_clock::now();
    double engineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "head pose time:" << std::endl;
    std::cout << engineTime << std::endl;
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


    for (float x : out){
	printf("%f \n",x);
	}

    //for (const auto& row : out) {
        // Loop through each element in the sub-vector
        //for (const auto& elem : row) {
        //    std::cout << elem << " ";
       // }
      //  std::cout << std::endl;  // New line for each row
    //}



	
   //outputQueue.push(out);

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


