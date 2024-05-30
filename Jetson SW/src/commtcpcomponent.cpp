#include "commtcpcomponent.h"
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <opencv2/opencv.hpp>
#include <thread>

// Constructor
CommTCPComponent::CommTCPComponent(int port, ThreadSafeQueue<cv::Mat>& outputQueue, ThreadSafeQueue<std::string>& commandsQueue, ThreadSafeQueue<std::string>& faultsQueue)
: port(port), outputQueue(outputQueue), commandsQueue(commandsQueue), faultsQueue(faultsQueue), running(false) {}

// Destructor
CommTCPComponent::~CommTCPComponent() {
    stopServer();
}

// Start server loop in a separate thread
void CommTCPComponent::startServer() {
    if (running) return;
    running = true;
    frameThread = std::thread(&CommTCPComponent::frameServerLoop, this);
    commandThread = std::thread(&CommTCPComponent::commandServerLoop, this);
    std::cout << "Server starting..." << std::endl;
}

// Release thread and any needed cleanup
void CommTCPComponent::stopServer() {
    running = false;
    if (frameThread.joinable()) {
        frameThread.join();
    }
    if (commandThread.joinable()) {
        commandThread.join();
    }
    std::cout << "Server stopped." << std::endl;
}

// Frame server loop
void CommTCPComponent::frameServerLoop() {
    int serverFd, newSocket;
    struct sockaddr_in address;
    int opt = 1;

    serverFd = socket(AF_INET, SOCK_STREAM, 0);
    fcntl(serverFd, F_SETFL, O_NONBLOCK);
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);  // Frame port

    bind(serverFd, (struct sockaddr*)&address, sizeof(address));
    listen(serverFd, 3);
    std::cout << "Frame server is ready and waiting for connections on port " << port << std::endl;

    while (running) {
        newSocket = accept(serverFd, NULL, NULL);
        if (newSocket > 0) {
            std::cout << "Client connected to frame server: socket FD " << newSocket << std::endl;
            std::thread(&CommTCPComponent::handleFrameClient, this, newSocket).detach();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    close(serverFd);
}

// Command server loop
void CommTCPComponent::commandServerLoop() {
    int serverFd, newSocket;
    struct sockaddr_in address;
    int opt = 1;

    serverFd = socket(AF_INET, SOCK_STREAM, 0);
    fcntl(serverFd, F_SETFL, O_NONBLOCK);
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port + 1);  // Command port, different from frame port

    bind(serverFd, (struct sockaddr*)&address, sizeof(address));
    listen(serverFd, 3);
    std::cout << "Command server is ready and waiting for connections on port " << (port + 1) << std::endl;

    while (running) {
        newSocket = accept(serverFd, NULL, NULL);
        if (newSocket > 0) {
            std::cout << "Client connected to command server: socket FD " << newSocket << std::endl;
            std::thread(&CommTCPComponent::handleCommandClient, this, newSocket).detach();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    close(serverFd);
}

// Handle frame transmission to a client
void CommTCPComponent::handleFrameClient(int clientSocket) {
    cv::Mat frame;
    while (running) {
        if (outputQueue.tryPop(frame) && !frame.empty()) {
            std::vector<uchar> buffer;
            cv::imencode(".jpg", frame, buffer);
            auto bufferSize = htonl(buffer.size()); // Convert to network byte order
            send(clientSocket, &bufferSize, sizeof(bufferSize), 0);
            send(clientSocket, buffer.data(), buffer.size(), 0);
        }
    }
    close(clientSocket);
}

// Handle command reception and other data transmission to a client
void CommTCPComponent::handleCommandClient(int clientSocket) {
    char buffer[1024] = {0};
    ssize_t bytesRead;

    const char* response = "test to sim readings from Jetson";
    send(clientSocket, response, strlen(response), 0);  // Correct usage
    while (running) {
        bytesRead = recv(clientSocket, buffer, sizeof(buffer), MSG_DONTWAIT);
        if (bytesRead > 0) {
            std::string command(buffer, bytesRead);
            std::cout << "Received command: " << command << std::endl;
            // Process command here and possibly send back a response
            // This part depends on the application logic
        }
    }
    close(clientSocket);
}

