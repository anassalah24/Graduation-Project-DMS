// BasicCameraComponent.cpp

#include "basiccameracomponent.h"

//constructor
BasicCameraComponent::BasicCameraComponent(ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
BasicCameraComponent::~BasicCameraComponent() {
    stopCapture();
}


//initialize function
bool BasicCameraComponent::initialize(const std::string& source) {
    if (!cap.open(source)) {
        std::cerr << "Failed to open camera or video file: " << source << std::endl;
        return false;
    }
    return true;
}


//This start the capture loop in another thread and then is destroyed ( still need to make sure of its destruction )
void BasicCameraComponent::startCapture() {
    if (running) {
        std::cerr << "Capture is already running." << std::endl;
        return;
    }
    running = true;
    captureThread = std::thread(&BasicCameraComponent::captureLoop, this);
}


//stop capture and release the thread
void BasicCameraComponent::stopCapture() {
    running = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (cap.isOpened()) {
        cap.release();
    }
}


// Main and most important loop that captures frame from camera or video file
void BasicCameraComponent::captureLoop() {
    fps = 60;
    //int delay = 1000 / fps;
    while (running) {
        // Calculate the delay based on the desired FPS
    	int delay = 1000 / (this->fps);  // Delay in milliseconds
        auto start = std::chrono::steady_clock::now();
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Failed to capture frame." << std::endl;
            running = false;
            break;
        }
        if (!frame.empty()) {
            outputQueue.push(frame);
        }

	// Calculate the time elapsed since the start of the loop
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // If the elapsed time is less than the desired delay, sleep for the remaining time
        if (elapsed < delay) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay - elapsed));
        }
    }
}

void BasicCameraComponent::setFPS(int fps) {
    this->fps = fps;
    std::cout << "FPS CHANGED SUCCESSFULLY" << std::endl;
}




