#pragma once

#include <thread>
#include <atomic>
#include "threadsafequeue.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

class CommTCPComponent {
public:
    CommTCPComponent(int port, ThreadSafeQueue<cv::Mat>& outputQueue, ThreadSafeQueue<std::string>& commandsQueue, ThreadSafeQueue<std::string>& faultsQueue);
    ~CommTCPComponent();

    void startServer();
    void stopServer();


private:
    int port;
    std::atomic<bool> running;
    std::thread frameThread;  // Thread handling frame transmissions
    std::thread commandThread;  // Thread handling command receptions and data transmissions
    ThreadSafeQueue<cv::Mat>& outputQueue;  // Queue for sending frames to connected client
    ThreadSafeQueue<std::string>& commandsQueue;  // Queue for processing commands
    ThreadSafeQueue<std::string>& faultsQueue;  // Queue for reporting faults

    // Separate server loops for frames and commands
    void frameServerLoop();  // Server loop for handling frame connections
    void commandServerLoop(); // Server loop for handling command connections

    // Individual client handlers for frames and commands
    void handleFrameClient(int clientSocket);  // Handles frame transmission to a client
    void handleCommandClient(int clientSocket); // Handles command reception from a client

    // Utility function to set up and configure a socket

};

