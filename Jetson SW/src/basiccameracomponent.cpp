#include "basiccameracomponent.h"

// Constructor
BasicCameraComponent::BasicCameraComponent(ThreadSafeQueue<cv::Mat>& outputQueue,
                                           ThreadSafeQueue<std::string>& commandsQueue,
                                           ThreadSafeQueue<std::string>& faultsQueue)
    : outputQueue(outputQueue), commandsQueue(commandsQueue), faultsQueue(faultsQueue), running(false) {}

// Destructor
BasicCameraComponent::~BasicCameraComponent() {
    stopCapture();
}

// Initialize function
bool BasicCameraComponent::initialize(const std::string& source) {
    if (!cap.open(source)) {
        std::cerr << "Failed to open camera or video file: " << source << std::endl;
        return false;
    }
    return true;
}

// Start the capture loop in another thread
void BasicCameraComponent::startCapture() {
    if (running) {
        std::cerr << "Capture is already running." << std::endl;
        return;
    }
    running = true;
    captureThread = std::thread(&BasicCameraComponent::captureLoop, this);
}

// Stop capture and release the thread
void BasicCameraComponent::stopCapture() {
    running = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
}

// Main loop that captures frame from camera or video file
void BasicCameraComponent::captureLoop() {
    //fps = 60;
    while (running) {
        int delay = 1000 / fps;
        auto start = std::chrono::steady_clock::now();
        
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Failed to capture frame." << std::endl;
            running = false;
            break;
        }
        

        if (!frame.empty()) {

            //int width = frame.cols;
            //int height = frame.rows;
            //size_t totalElements = frame.total();
            //int dataTypeSize = frame.elemSize();  
            //size_t frameSize = totalElements * dataTypeSize;

            outputQueue.push(frame);
        }

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


        if (elapsed < delay) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay - elapsed));
    	    std::cout << "SLEPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" << std::endl;
        }
    }
    running = false;
}

// Set FPS
void BasicCameraComponent::setFPS(int fps) {
    this->fps = fps;
    std::cout << "FPS changed successfully" << std::endl;
}

// Set the source and restart capture
void BasicCameraComponent::setSource(const std::string& source) {
    stopCapture();
    cap.release() ;
    if (!cap.open(source)) {
        std::cerr << "Failed to change source: " << source << std::endl;
    } else {
        startCapture();
        std::cout << "Changed source to: " << source << std::endl;
    }
}

