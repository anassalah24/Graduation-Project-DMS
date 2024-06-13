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
#include <cstdint>
#include <cstring>  // for memcpy


// Constructor
CommTCPComponent::CommTCPComponent(int port, ThreadSafeQueue<cv::Mat>& outputQueue, ThreadSafeQueue<std::vector<std::vector<float>>>& readingsQueue, ThreadSafeQueue<std::string>& commandsQueue, ThreadSafeQueue<std::string>& faultsQueue)
: port(port), outputQueue(outputQueue), readingsQueue(readingsQueue), commandsQueue(commandsQueue), faultsQueue(faultsQueue), running(false) {}

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

// Handle command reception and readings data transmission to a client
void CommTCPComponent::handleCommandClient(int clientSocket) {
    std::vector<std::vector<float>> reading;

    while (running) {
        // Handle readings data transmission
        if (readingsQueue.tryPop(reading) && !reading.empty()) {
            std::vector<uint8_t> serializedData = serialize(reading);
            send(clientSocket, serializedData.data(), serializedData.size(), 0);
        }

        // Handle configuration messages
        char buffer[1024] = {0};
        ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer), MSG_DONTWAIT);

        if (bytesRead > 0) {
            const char* ptr = buffer;
            const char* end = buffer + bytesRead;

            while (ptr < end) {
                // Read the length of the next message
                int messageLength = 0;
                std::memcpy(&messageLength, ptr, sizeof(int));
                ptr += sizeof(int);

                // Ensure we have enough data for the message
                if (ptr + messageLength > end) {
                    std::cerr << "Incomplete message received" << std::endl;
                    break;
                }

                // Extract the message
                std::string message(ptr, messageLength);
                ptr += messageLength;

                // Process the message
                if (message.find("SET_FPS") != std::string::npos) {
                    // Handle FPS configuration
                    std::cout << "Received SET_FPS command with value: " << message.substr(8) << std::endl;
                    std::string command = "SET_FPS:" + message.substr(8);
                    commandsQueue.push(command);

                } else if (message == "TURN_OFF") {
                    // Handle turning off the system
                    std::cout << "Received TURN_OFF command" << std::endl;
                    std::string command = "TURN_OFF";
                    commandsQueue.push(command);

                } else if (message == "TURN_ON") {
                    // Handle turning on the system
                    std::cout << "Received TURN_ON command" << std::endl;
                    std::string command = "TURN_ON";
                    commandsQueue.push(command);

                } else if (message.find("SET_FDT") != std::string::npos) {
                    // Handle FDT configuration
                    std::cout << "Received SET_FDT command with value: " << message.substr(8) << std::endl;
                    std::string command = "SET_FDT:" + message.substr(8);
                    commandsQueue.push(command);

		} else if (message.find("SET_SOURCE") != std::string::npos) {
                    // Handle FDT configuration
                    std::cout << "Received SET_SOURCE command with value: " << message.substr(11) << std::endl;
                    std::string command = "SET_SOURCE:" + message.substr(11);
                    commandsQueue.push(command);

                } else if (message.find("SET_FD_MODEL") != std::string::npos) {
                    std::cout << "Received SET_FD_MODEL command with value: " << message.substr(13) << std::endl;
                    commandsQueue.push("SET_FD_MODEL:" + message.substr(13));

                } else if (message.find("SET_HP_MODEL") != std::string::npos) {
                    std::cout << "Received SET_HP_MODEL command with value: " << message.substr(13) << std::endl;
                    commandsQueue.push("SET_HP_MODEL:" + message.substr(13));

                } else if (message.find("SET_EG_MODEL") != std::string::npos) {
                    std::cout << "Received SET_EG_MODEL command with value: " << message.substr(13) << std::endl;
                    commandsQueue.push("SET_EG_MODEL:" + message.substr(13));

                } else {
                    std::cout << "Received unknown command: " << message << std::endl;
                }
            }
        }
    }

    close(clientSocket);
}
// Serialize a 2D vector of floats to a byte array
std::vector<uint8_t> CommTCPComponent::serialize(const std::vector<std::vector<float>>& data) {
    std::vector<uint8_t> buffer;
    size_t rows = data.size();
    size_t cols = rows > 0 ? data[0].size() : 0;

    // Add size information
    buffer.resize(sizeof(size_t) * 2 + rows * cols * sizeof(float));
    uint8_t* ptr = buffer.data();

    // Copy rows and cols to the buffer
    std::memcpy(ptr, &rows, sizeof(size_t));
    ptr += sizeof(size_t);
    std::memcpy(ptr, &cols, sizeof(size_t));
    ptr += sizeof(size_t);

    // Copy each float value to the buffer
    for (const auto& row : data) {
        std::memcpy(ptr, row.data(), row.size() * sizeof(float));
        ptr += row.size() * sizeof(float);
    }

    return buffer;
}

