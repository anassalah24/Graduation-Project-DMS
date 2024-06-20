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
    // Constructor
    CommTCPComponent(int port, ThreadSafeQueue<cv::Mat>& outputQueue, 
                     ThreadSafeQueue<std::vector<std::vector<float>>>& readingsQueue, 
                     ThreadSafeQueue<std::string>& commandsQueue, 
                     ThreadSafeQueue<std::string>& faultsQueue);
    
    // Destructor
    ~CommTCPComponent();

    // Start the TCP server
    void startServer();
    
    // Stop the TCP server
    void stopServer();

    // Data transfer metrics getters
    size_t getTotalFrameDataSent() const { return totalFrameDataSent; }
    size_t getTotalCommandDataSent() const { return totalCommandDataSent; }
    size_t getTotalReadingsDataSent() const { return totalReadingsDataSent; }
    size_t getFrameCount() const { return frameCount; }
    size_t getTransmissionErrors() const { return transmissionErrors; }

    // Reset data transfer metrics
    void resetDataTransferMetrics() {
        totalFrameDataSent = 0;
        totalCommandDataSent = 0;
        totalReadingsDataSent = 0;
        frameCount = 0;
        transmissionErrors = 0;
    }

    // Log data transfer metrics
    void logDataTransferMetrics();

private:
    int port;
    std::atomic<bool> running;
    std::thread frameThread;  // Thread handling frame transmissions
    std::thread commandThread;  // Thread handling command receptions and data transmissions
    ThreadSafeQueue<cv::Mat>& outputQueue; // Queue for sending frames to connected clients
    ThreadSafeQueue<std::vector<std::vector<float>>>& readingsQueue; // Queue for sending readings to connected clients
    ThreadSafeQueue<std::string>& commandsQueue;  // Queue for processing commands
    ThreadSafeQueue<std::string>& faultsQueue;  // Queue for reporting faults

    size_t totalFrameDataSent = 0;
    size_t totalCommandDataSent = 0;
    size_t totalReadingsDataSent = 0;
    size_t frameCount = 0;
    size_t transmissionErrors = 0;

    // Separate server loops for frames and commands
    void frameServerLoop();  // Server loop for handling frame connections
    void commandServerLoop(); // Server loop for handling command connections

    // Individual client handlers for frames and commands
    void handleFrameClient(int clientSocket);  // Handles frame transmission to a client
    void handleCommandClient(int clientSocket); // Handles command reception from a client

    // Serialize a 2D vector of floats to a byte array
    std::vector<uint8_t> serialize(const std::vector<std::vector<float>>& data);
};

