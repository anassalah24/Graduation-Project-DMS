#pragma once

#include <thread>
#include <signal.h>
#include "threadsafequeue.h"
#include "basiccameracomponent.h"
#include "facedetectioncomponent.h"
#include "aicomponent.h"
#include "commtcpcomponent.h"




class DMSManager {
public:
    DMSManager(ThreadSafeQueue<cv::Mat>& cameraQueue,
               ThreadSafeQueue<cv::Mat>& faceDetectionQueue, ThreadSafeQueue<cv::Rect>& faceRectQueue,
               ThreadSafeQueue<std::vector<std::vector<float>>>& AIDetectionQueue, 
               ThreadSafeQueue<cv::Mat>& framesQueue, 
               ThreadSafeQueue<cv::Mat>& tcpOutputQueue, 
               int tcpPort,
               ThreadSafeQueue<std::string>& commandsQueue, ThreadSafeQueue<std::string>& faultsQueue);
    ~DMSManager();

    bool startSystem();
    void stopSystem();
    bool initializeCamera(const std::string& source);
    void setCameraFPS(int fps);
    void setFaceFDT(int fdt);
    void setCamereSource(const std::string& source);
    void clearQueues();
    void setupSignalHandlers();

    // Function to handle the different types of commands 
    void handleCommand(std::string& command);

private:
    BasicCameraComponent cameraComponent;
    FaceDetectionComponent faceDetectionComponent;
    AIComponent AiComponent;
    CommTCPComponent tcpComponent; 

    ThreadSafeQueue<cv::Mat>& cameraQueue;
    ThreadSafeQueue<cv::Mat>& faceDetectionQueue;
    ThreadSafeQueue<cv::Rect>& faceRectQueue;
    ThreadSafeQueue<std::vector<std::vector<float>>>& AIDetectionQueue;
    ThreadSafeQueue<cv::Mat>& framesQueue;
    ThreadSafeQueue<cv::Mat>& tcpOutputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;

    std::thread cameraThread;
    std::thread faceDetectionThread;
    std::thread AIThread;
    std::thread tcpThread; 
    std::thread commandsThread; 

    int tcpPort; 
    bool running;
    bool firstRun = true;

    // Component loops that start in their own thread
    void cameraLoop();
    void faceDetectionLoop();
    void AILoop();
    void commtcpLoop(); 
    void commandsLoop();
};

