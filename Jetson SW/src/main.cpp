#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include "dmsmanager.h"
#include <benchmark/benchmark.h>


int main() {


    // Initialize thread-safe queues needed for each component
    ThreadSafeQueue<cv::Mat> cameraQueue;
    ThreadSafeQueue<cv::Mat> preprocessingQueue;
    ThreadSafeQueue<cv::Mat> faceDetectionQueue; 
    ThreadSafeQueue<cv::Rect> faceRectQueue;
    ThreadSafeQueue<cv::Mat> drowsinessDetectionQueue;
    ThreadSafeQueue<std::vector<std::vector<float>>> headposeDetectionQueue;
    ThreadSafeQueue<std::string> eyegazeDetectionQueue;
    ThreadSafeQueue<cv::Mat> framesQueue;
    ThreadSafeQueue<cv::Mat> eyegazeframesQueue;
    ThreadSafeQueue<cv::Mat> tcpOutputQueue;
    ThreadSafeQueue<CarState> stateOutputQueue;
    ThreadSafeQueue<int> postOutputQueue;
    ThreadSafeQueue<std::string> commandsQueue;
    ThreadSafeQueue<std::string> faultsQueue;

    int tcpPort = 12345;  // Define the TCP port for the server

    // Initialize the DMSManager with all necessary queues and the TCP port
    DMSManager dmsManager(cameraQueue, preprocessingQueue, faceDetectionQueue, faceRectQueue,
 drowsinessDetectionQueue, headposeDetectionQueue,eyegazeDetectionQueue,framesQueue,eyegazeframesQueue, tcpOutputQueue, tcpPort, stateOutputQueue, postOutputQueue,commandsQueue,faultsQueue);

    // Initialize the camera component
///home/dms/Downloads/drowsiness_detection-master/src/sample_videos/driver_day.mp4
    if (!dmsManager.initializeCamera("/dev/video0")) { // use /dev/video0 for camera or /path/to/video/file
        std::cerr << "Failed to initialize camera component." << std::endl;
        return -1;
    }

    // Initialize the face detection component with weights and configurations
    if (!dmsManager.initializeFaceDetection("/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.cfg", "/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.weights")) {
        std::cerr << "Failed to initialize face detection component." << std::endl;
        return -1;
    }


    //****************** Initialize the drowsiness detection component with anything needed
    //if (!dmsManager.initializeDrowsinessDetection()) {
    //    std::cerr << "Failed to initialize dowsiness detection component." << std::endl;
    //    return -1;
    //}

    //****************** Initialize the head pose detection component with anything needed
    if (!dmsManager.initializeHeadposeDetection()) {
        std::cerr << "Failed to initialize headpose detection component." << std::endl;
        return -1;
    }

    //******************missing initialization for the vehicle state to pass the file to read from***************************
    //-----------------------------------------------------------------------------------------------------------------------------


    // Start the system
    if (!dmsManager.startSystem()) {
        std::cerr << "Failed to start the system." << std::endl;
        return -1;
    }


    // Main loop (The display code is commented out; uncomment if needed)

    
    // MAT objects to carry frames incase we need to show frames of some output from component while development
    cv::Mat cameraFrame, preprocessedFrame, faceDetectedFrame, drowsyDetectedFrame,headposeFrame,eyegazeFrame;
    while (true) {
        //if (cameraQueue.tryPop(cameraFrame) && !cameraFrame.empty()) {
        //   cv::imshow("Camera Frame", cameraFrame);
        //}

        //if (preprocessingQueue.tryPop(preprocessedFrame) && !preprocessedFrame.empty()) {
        //    cv::imshow("Processed Frame", preprocessedFrame);
        //}

        //if (faceDetectionQueue.tryPop(faceDetectedFrame) && !faceDetectedFrame.empty()) {
        //   cv::imshow("Face Detection", faceDetectedFrame);
        //}

	    //if (drowsinessDetectionQueue.tryPop(drowsyDetectedFrame) && !drowsyDetectedFrame.empty()) {
        //   cv::imshow("drowsy Detection", drowsyDetectedFrame);
        //}

        if (framesQueue.tryPop(headposeFrame) && !headposeFrame.empty()) {
           cv::imshow("Head Pose", headposeFrame);
        }

        //if (eyegazeframesQueue.tryPop(eyegazeFrame) && !eyegazeFrame.empty()) {
        //   cv::imshow("Eye Gaze", eyegazeFrame);
        //}


        if (cv::waitKey(1) == 27) break;  // Exit on ESC key
    }

    // Shutdown the system
    dmsManager.stopSystem();
    std::cout << "System stopped." << std::endl;


    return 0;
}

