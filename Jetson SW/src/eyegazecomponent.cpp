#include "eyegazecomponent.h"
#include "inferC.h"


TRTEngineSingletonC* TRTEngineSingletonC::instance = nullptr;

//constructor
EyeGazeComponent::EyeGazeComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputFramesQueue, ThreadSafeQueue<std::string>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputFramesQueue(outputFramesQueue), outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
EyeGazeComponent::~EyeGazeComponent() {
    stopEyeGazeDetection();
}

// initialize model 
bool EyeGazeComponent::initialize() {
    return true;
}

// start detection loop in another thread
void EyeGazeComponent::startEyeGazeDetection() {
    if (running) {
        std::cerr << "eyegaze Detection is already running." << std::endl;
        return;
    }
    running = true;
    EyeGazeDetectionThread = std::thread(&EyeGazeComponent::EyeGazeDetectionLoop, this);
}

// release thread and any needed cleanup
void EyeGazeComponent::stopEyeGazeDetection() {
    running = false;
    if (EyeGazeDetectionThread.joinable()) {
        EyeGazeDetectionThread.join();
    }
}


// This loop takes frame from input queue , sends it to detect eyegaze and places it into the output queue
void EyeGazeComponent::EyeGazeDetectionLoop() {
    cv::Mat frame;
    this->lastTime = std::chrono::high_resolution_clock::now(); // Initialize the last time
    bool isFirstFrame = true; // Flag to check if it's the first frame
    while (running) {
        if (inputQueue.tryPop(frame)) {

            //std::cout << "about to start eye gaze" << std::endl;

            //auto start = std::chrono::high_resolution_clock::now();
            detectEyeGaze(frame);
            //auto end = std::chrono::high_resolution_clock::now();

            //double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            //updatePerformanceMetrics(detectionTime);
            //displayPerformanceMetrics(frame);

            outputFramesQueue.push(frame);

	    //if (isFirstFrame) {
                //inputQueue.clear(); // Clear the queue after the first frame is processed
                //isFirstFrame = false; // Update the flag so the queue won't be cleared again
            //}


        }
    }
}






//function to start the head pose detection
void EyeGazeComponent::detectEyeGaze(cv::Mat& frame) {

    TRTEngineSingletonC* trt=TRTEngineSingletonC::getInstance();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> out = trt->infer(frame);
    auto end = std::chrono::high_resolution_clock::now();
    double engineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout << "Eye gaze time:" << std::endl;
    //std::cout << engineTime << std::endl;
    std::string timeText = "time: " + std::to_string(int(engineTime))+"ms";
    cv::putText(frame, timeText, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    double fps = 1000/(engineTime+120+35);
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    int textPositionY = 55;  // Initial Y position for the first text

    for (float x : out) {
	    std::string text = std::to_string(x);  // Convert the float to string

	    // Put the text on the image
	    cv::putText(frame, text, cv::Point(20, textPositionY), 
		        cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1);

	    textPositionY += 15;  // Move down for the next text
	    if (textPositionY > frame.rows - 15) {  // Check if the position is out of image bounds
		break;  // Reset position or break if you don't want to overwrite
	    }
    }
    //for (float x : out){
	//printf("%f \n",x);
	//}
	//
   //outputQueue.push(out);

}

//



void EyeGazeComponent::updatePerformanceMetrics(double detectionTime) {
    totalDetectionTime += detectionTime;  // Total time spent on detection
    totalFramesProcessed++;               // Increment the frame count

    // Calculation of average time per frame and updating fps immediately
    avgDetectionTime = totalDetectionTime / totalFramesProcessed;
    auto currentTime = std::chrono::high_resolution_clock::now();
    fps = totalFramesProcessed / (std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count() + 1e-9);  // Adding a small epsilon to avoid division by zero
}

void EyeGazeComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time per Frame: " + std::to_string(avgDetectionTime) + " ms";

    // Output the metrics to the console; you might want to output to the image or a GUI in a real application
    //std::cout << fpsText << std::endl;
    //std::cout << avgTimeText << std::endl;
}


