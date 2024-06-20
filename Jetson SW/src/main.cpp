#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include "dmsmanager.h"
#include <benchmark/benchmark.h>

int main() {

    // Initialize thread-safe queues needed for each component
    ThreadSafeQueue<cv::Mat> cameraQueue;
    ThreadSafeQueue<cv::Mat> faceDetectionQueue; 
    ThreadSafeQueue<cv::Rect> faceRectQueue;
    ThreadSafeQueue<std::vector<std::vector<float>>> AIDetectionQueue;
    ThreadSafeQueue<cv::Mat> framesQueue;
    ThreadSafeQueue<cv::Mat> tcpOutputQueue;
    ThreadSafeQueue<std::string> commandsQueue;
    ThreadSafeQueue<std::string> faultsQueue;

    int tcpPort = 12345;  // Define the TCP port for the server

    // Initialize the DMSManager with all necessary queues and the TCP port
    DMSManager dmsManager(cameraQueue, faceDetectionQueue, faceRectQueue,
                          AIDetectionQueue,framesQueue,
			  tcpOutputQueue, tcpPort,
                          commandsQueue, faultsQueue);

    // Initialize the camera component
    if (!dmsManager.initializeCamera("/dev/video0")) { // use /dev/video0 for camera or /path/to/video/file
        std::cerr << "Failed to initialize camera component." << std::endl;
        return -1;
    }

    // Start the system
    if (!dmsManager.startSystem()) {
        std::cerr << "Failed to start the system." << std::endl;
        return -1;
    }

    // Main loop (The display code is commented out; uncomment if needed)
    cv::Mat cameraFrame;
    while (true) {

        // Uncomment the following block to show frames from different components during development , if need to show frames on jetson

        // if (cameraQueue.tryPop(cameraFrame) && !cameraFrame.empty()) {
        //    cv::imshow("Camera Frame", cameraFrame);
        // }

        if (cv::waitKey(1) == 27) break;  // Exit on ESC key
    }

    // Shutdown the system
    dmsManager.stopSystem();
    std::cout << "System stopped." << std::endl;

    return 0;
}

