// BasicPreprocessingComponent.cpp

#include "basicpreprocessingcomponent.h"


//constructor
BasicPreprocessingComponent::BasicPreprocessingComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue,
ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue),running(false) {}

BasicPreprocessingComponent::~BasicPreprocessingComponent() {
    stopProcessing();
}


//destructor
void BasicPreprocessingComponent::startProcessing() {
    if (running) {
        std::cerr << "Processing is already running." << std::endl;
        return;
    }
    running = true;
    processingThread = std::thread(&BasicPreprocessingComponent::processingLoop, this);
}

//function to release the thread ( add any cleanup needed )
void BasicPreprocessingComponent::stopProcessing() {
    running = false;
    if (processingThread.joinable()) {
        processingThread.join();
    }
}

// loop that takes input frame coming from camera and sends it to process frame and them place them into the output queue
void BasicPreprocessingComponent::processingLoop() {
    cv::Mat frame;
    while (running) {
        if (inputQueue.tryPop(frame)) {
            cv::Mat processedFrame = preprocessFrame(frame);
            if (!processedFrame.empty()) {
                outputQueue.push(processedFrame);
            }
        }
    }
}


// main preprocessing function ( contains till now resizing frame only )
cv::Mat BasicPreprocessingComponent::preprocessFrame(const cv::Mat& frame) {

  cv::Mat processedFrame;

  // Define the desired crop size
  int crop_width = 350; // Replace with your desired width
  int crop_height = 350; // Replace with your desired height

  // Get the original image height and width
  int original_height = frame.rows;
  int original_width = frame.cols;

  // Calculate the starting coordinates for centering the crop
  int start_x = (original_width - crop_width) / 2;
  int start_y = (original_height - crop_height) / 2;

  // Perform cropping using ROI (Region of Interest)
  cv::Rect roi(start_x, start_y, crop_width, crop_height);
  processedFrame = frame(roi);  // Extract the ROI into processedFrame

  // Add more preprocessing steps as needed (e.g., grayscale conversion)

  return processedFrame;
}

//********************************missing dynamically extracting the ROI**************************************
//---------------------------------------------------------------------------------------------------------




