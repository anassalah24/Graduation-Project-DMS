#include "dmsmanager.h"
#include <benchmark/benchmark.h>

// Constructor: passes input and output queues for different components
DMSManager::DMSManager(ThreadSafeQueue<cv::Mat>& cameraQueue, 
                       ThreadSafeQueue<cv::Mat>& faceDetectionQueue, 
                       ThreadSafeQueue<cv::Rect>& faceRectQueue,
                       ThreadSafeQueue<std::vector<std::vector<float>>>& AIDetectionQueue, 
                       ThreadSafeQueue<cv::Mat>& framesQueue, 
                       ThreadSafeQueue<cv::Mat>& tcpOutputQueue, 
                       int tcpPort, 
                       ThreadSafeQueue<std::string>& commandsQueue,
                       ThreadSafeQueue<std::string>& faultsQueue)
    : cameraComponent(cameraQueue, commandsQueue, faultsQueue),
      faceDetectionComponent(cameraQueue, faceDetectionQueue, faceRectQueue, commandsQueue, faultsQueue),
      AiComponent(faceDetectionQueue, faceRectQueue, AIDetectionQueue, framesQueue, commandsQueue, faultsQueue),
      tcpComponent(tcpPort, framesQueue, AIDetectionQueue, commandsQueue, faultsQueue),
      cameraQueue(cameraQueue), 
      faceDetectionQueue(faceDetectionQueue), 
      faceRectQueue(faceRectQueue),
      AIDetectionQueue(AIDetectionQueue),
      framesQueue(framesQueue), 
      tcpPort(tcpPort), 
      tcpOutputQueue(tcpOutputQueue), 
      commandsQueue(commandsQueue),
      faultsQueue(faultsQueue),
      running(false), 
      firstRun(true) {}

// Destructor (cleanup)
DMSManager::~DMSManager() {
    stopSystem();
}

// Startup system
bool DMSManager::startSystem() {
    if (running) return false;  // Prevent the system from starting if it's already running
    running = true;

    if (firstRun) {
        // Starting each component in its own thread
        cameraThread = std::thread(&DMSManager::cameraLoop, this);  // Start the camera loop in its own thread
        faceDetectionThread = std::thread(&DMSManager::faceDetectionLoop, this);  // Start face detection in its own thread
        AIThread = std::thread(&DMSManager::AILoop, this);  // Start AI detection in its own thread
        tcpThread = std::thread(&DMSManager::commtcpLoop, this); // Start tcp thread in its own thread
        commandsThread = std::thread(&DMSManager::commandsLoop, this); // Start commands thread in its own thread
        firstRun = false;
        return true;
    } else {
        // restart threads
        cameraThread = std::thread(&DMSManager::cameraLoop, this);
        faceDetectionThread = std::thread(&DMSManager::faceDetectionLoop, this);
        AIThread = std::thread(&DMSManager::AILoop, this);
        firstRun = false;
        return true;
    }
}

// Stop the system
void DMSManager::stopSystem() {
    running = false;  // Signal all loops to stop
    clearQueues();

    // Log performance metrics
    AiComponent.logPerformanceMetrics();
    tcpComponent.logDataTransferMetrics();
    faceDetectionComponent.logPerformanceMetrics();

    // Stop components
    cameraComponent.stopCapture();
    faceDetectionComponent.stopDetection();
    AiComponent.stopAIDetection();

    // Wait for each thread to finish its task and join
    if (cameraThread.joinable()) cameraThread.join();
    if (faceDetectionThread.joinable()) faceDetectionThread.join();
    if (AIThread.joinable()) AIThread.join();


}

// Setup signal handlers
void DMSManager::setupSignalHandlers() {
    signal(SIGPIPE, SIG_IGN);
}

// Component loops that start in their own thread
void DMSManager::cameraLoop() {
    cameraComponent.startCapture();
}

void DMSManager::faceDetectionLoop() {
    faceDetectionComponent.startDetection();
}

void DMSManager::AILoop() {
    AiComponent.startAIDetection();
}

void DMSManager::commtcpLoop() {
    setupSignalHandlers();
    tcpComponent.startServer();
}

// Loop for DMSManager component to check for any needed commands by components
void DMSManager::commandsLoop(){
    std::string command;
    while (true) {
        if (commandsQueue.tryPop(command)) {
            std::cout << "Received command in the DMS manager: " << command << std::endl;
            this->handleCommand(command);
        }
    }
}

// Functions for different configurations
void DMSManager::setCameraFPS(int fps) {
    cameraComponent.setFPS(fps);
}

void DMSManager::setFaceFDT(int fdt) {
    faceDetectionComponent.setFDT(fdt);
}

void DMSManager::setCamereSource(const std::string& source) {
    cameraComponent.setSource(source);
    clearQueues();
}

void DMSManager::clearQueues(){
    cameraQueue.clear();
    faceDetectionQueue.clear();
    faceRectQueue.clear();
    AIDetectionQueue.clear();
    framesQueue.clear();
    tcpOutputQueue.clear();
}

void DMSManager::handleCommand(std::string& command) {
    clearQueues();
    //model paths
    std::map<std::string, std::string> headPoseModels = {
        {"AX", "/home/dms/DMS/ModularCode/include/Ax.engine"},
        {"AY", "/home/dms/DMS/ModularCode/include/Ay.engine"},
        {"AZ", "/home/dms/DMS/ModularCode/include/Az.engine"},
        {"A0", "/home/dms/DMS/ModularCode/include/A0.engine"},
        {"eff0", "/home/dms/DMS/ModularCode/include/eff0.engine"},
        {"eff1", "/home/dms/DMS/ModularCode/include/eff1.engine"},
        {"eff2", "/home/dms/DMS/ModularCode/include/eff2.engine"},
        {"eff3", "/home/dms/DMS/ModularCode/include/eff3.engine"},
        {"whenNet", "/home/dms/DMS/ModularCode/include/whenNet.engine"},
        {"No Head Pose", "No Head Pose"}
    };

    std::map<std::string, std::string> eyeGazeModels = {
        {"mobilenetv3", "/home/dms/DMS/ModularCode/modelconfigs/mobilenetv3_engine.engine"},
        {"squeezenet", "/home/dms/DMS/ModularCode/modelconfigs/squeezenet.engine"},
        {"resnet", "/home/dms/DMS/ModularCode/include/resnet_engine.engine"},
        {"mobilenet", "/home/dms/DMS/ModularCode/include/mobilenet_engine.engine"},
        {"No Eye Gaze", "No Eye Gaze"}
    };

    std::map<std::string, std::pair<std::string, std::string>> faceDetectionModels = {
        {"YoloV3 Tiny", {"/home/dms/DMS/ModularCode/modelconfigs/face-yolov3-tiny.cfg", "/home/dms/DMS/ModularCode/modelconfigs/face-yolov3-tiny_41000.weights"}},
        {"YoloV2", {"/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.cfg", "/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.weights"}},
        {"No Face Detection", {"No Face Detection", "No Face Detection"}}

    };

    // Setting FPS
    if (command.find("SET_FPS:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string fpsValueStr = command.substr(pos + 1);
            int fpsValue = std::stoi(fpsValueStr);
            std::cout << "Setting FPS to: " << fpsValue << std::endl;
            setCameraFPS(fpsValue);
        } else {
            std::cerr << "Invalid SETFPS command format: " << command << std::endl;
        }
    }
    // Setting face detection threshold
    else if (command.find("SET_FDT:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string fdtValueStr = command.substr(pos + 1);
            int fdtValue = std::stoi(fdtValueStr);
            std::cout << "Setting Face Detection Threshold to: " << fdtValue << std::endl;
            setFaceFDT(fdtValue);
        } else {
            std::cerr << "Invalid SETFDT command format: " << command << std::endl;
        }
    }
    // Setting source
    else if (command.find("SET_SOURCE:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string sourceStr = command.substr(pos + 1);
            if (sourceStr == "camera") {
                sourceStr = "/dev/video0";  // or the appropriate camera source for your system
            } else if (sourceStr.find("video:") == 0) {
                std::string videoName = sourceStr.substr(6);  // Extract the video name after "video:"
                sourceStr = "/home/dms/DMS/Videos/" + videoName;  // Construct the full path to the video file
            } else {
                std::cerr << "Invalid source value: " << sourceStr << std::endl;
                return;
            }
            std::cout << "Setting Source to: " << sourceStr << std::endl;
            setCamereSource(sourceStr);
        } else {
            std::cerr << "Invalid SET_SOURCE command format: " << command << std::endl;
        }
    }
    // Turning off the system
    else if (command == "TURN_OFF") {
        std::cout << "Turning off..." << std::endl;
        stopSystem();
    }
    // Turning on the system
    else if (command == "TURN_ON") {
        std::cout << "Turning on..." << std::endl;
        startSystem();
    }
    // Clear Queue
    else if (command == "Clear Queue") {
        clearQueues();
    }
    // Handling Face Detection Model
    else if (command.find("SET_FD_MODEL:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string modelValue = command.substr(pos + 1);
            std::cout << "Setting Face Detection Model to: " << modelValue << std::endl;
            auto it = faceDetectionModels.find(modelValue);
            if (it != faceDetectionModels.end()) {
                std::string weightPath = it->second.second;
                std::string configPath = it->second.first;
                if (weightPath == "No Face Detection" && configPath == "No Face Detection") {
                    faceDetectionComponent.modelstatus = false;
                    std::cout << "Updated Face Detection Model and Config to: " << weightPath << " and " << configPath << std::endl;
                    clearQueues();
                } else {
                    faceDetectionComponent.stopDetection();
                    faceDetectionComponent.initialize(configPath, weightPath);
                    faceDetectionComponent.modelstatus = true;
                    faceDetectionComponent.startDetection();
                    std::cout << "Updated Face Detection Model and Config to: " << weightPath << " and " << configPath << std::endl;
                    clearQueues();
                }
            } else {
                std::cerr << "Face detection model identifier not recognized: " << modelValue << std::endl;
            }
        } else {
            std::cerr << "Invalid SET_FD_MODEL command format: " << command << std::endl;
        }
    }
    // Handling Head Pose Model
    else if (command.find("SET_HP_MODEL:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string modelValue = command.substr(pos + 1);
            std::cout << "Setting Head Pose Model to: " << modelValue << std::endl;
            if (headPoseModels.find(modelValue) != headPoseModels.end()) {
                clearQueues();
                AiComponent.updateHeadPoseEngine(headPoseModels[modelValue]);
                clearQueues();
            } else {
                std::cerr << "Head pose model identifier not recognized: " << modelValue << std::endl;
            }
        } else {
            std::cerr << "Invalid SET_HP_MODEL command format: " << command << std::endl;
        }
    }
    // Handling Eye Gaze Model
    else if (command.find("SET_EG_MODEL:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string modelValue = command.substr(pos + 1);
            std::cout << "Setting Eye Gaze Model to: " << modelValue << std::endl;
            if (eyeGazeModels.find(modelValue) != eyeGazeModels.end()) {
                clearQueues();
                AiComponent.updateEyeGazeEngine(eyeGazeModels[modelValue]);
                clearQueues();
            } else {
                std::cerr << "Eye gaze model identifier not recognized: " << modelValue << std::endl;
            }
        } else {
            std::cerr << "Invalid SET_EG_MODEL command format: " << command << std::endl;
        }
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
    }
}

// Initialization functions needed for some components
bool DMSManager::initializeCamera(const std::string& source) {
    return cameraComponent.initialize(source);
}




