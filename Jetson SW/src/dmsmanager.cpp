#include "dmsmanager.h"
#include <benchmark/benchmark.h>

//Constructor ; passes input and ouptut queues for different component and tcp port
DMSManager::DMSManager(ThreadSafeQueue<cv::Mat>& cameraQueue, ThreadSafeQueue<cv::Mat>& preprocessingQueue, ThreadSafeQueue<cv::Mat>& faceDetectionQueue, ThreadSafeQueue<cv::Rect>& faceRectQueue,
ThreadSafeQueue<cv::Mat>& drowsinessDetectionQueue,ThreadSafeQueue<std::vector<std::vector<float>>>& headposeDetectionQueue, ThreadSafeQueue<std::string>& eyegazeDetectionQueue,ThreadSafeQueue<cv::Mat>& framesQueue, ThreadSafeQueue<cv::Mat>& eyegazeframesQueue, ThreadSafeQueue<cv::Mat>& tcpOutputQueue, int tcpPort , ThreadSafeQueue<CarState>& stateOutputQueue , ThreadSafeQueue<int>& postOutputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
:
cameraComponent(cameraQueue,commandsQueue,faultsQueue),
preprocessingComponent(cameraQueue, preprocessingQueue,commandsQueue,faultsQueue), 
faceDetectionComponent(cameraQueue, faceDetectionQueue, faceRectQueue, commandsQueue,faultsQueue),
drowsinessComponent(faceDetectionQueue, drowsinessDetectionQueue,commandsQueue,faultsQueue),
headposeComponent(faceDetectionQueue, faceRectQueue, headposeDetectionQueue,framesQueue,commandsQueue,faultsQueue),
eyegazeComponent(framesQueue, eyegazeframesQueue, eyegazeDetectionQueue,commandsQueue,faultsQueue),
tcpComponent(tcpPort, framesQueue, headposeDetectionQueue, commandsQueue,faultsQueue),
vehicleStateManager(stateOutputQueue,commandsQueue,faultsQueue),
postProcessingComponent(stateOutputQueue, postOutputQueue,commandsQueue,faultsQueue),
faultManager(commandsQueue, faultsQueue),
cameraQueue(cameraQueue),preprocessingQueue(preprocessingQueue), faceDetectionQueue(faceDetectionQueue), faceRectQueue(faceRectQueue),
drowsinessDetectionQueue(drowsinessDetectionQueue),headposeDetectionQueue(headposeDetectionQueue),
eyegazeDetectionQueue(eyegazeDetectionQueue), framesQueue(framesQueue), eyegazeframesQueue(eyegazeframesQueue),
tcpPort(tcpPort), tcpOutputQueue(tcpOutputQueue), stateOutputQueue(stateOutputQueue),postOutputQueue(postOutputQueue),
commandsQueue(commandsQueue),faultsQueue(faultsQueue),running(false) {}//change back the tcp queue

//destructor (cleanup)
DMSManager::~DMSManager() {
    stopSystem();
}


//startup system
bool DMSManager::startSystem() {
    if (running) return false;  // Prevent the system from starting if it's already running
    running = true;

	if (firstRun){
	    //starting each component in its own thread
	    cameraThread = std::thread(&DMSManager::cameraLoop, this);  // Start the camera loop in its own thread
	    //preprocessingThread = std::thread(&DMSManager::preprocessingLoop, this);  // Start the preprocessing loop in its own thread
	    faceDetectionThread = std::thread(&DMSManager::faceDetectionLoop, this);  // Start face detection in its own thread
	    //drowsinessThread = std::thread(&DMSManager::drowsinessLoop, this);  // Start drowsiness detection in its own thread
	    headposeThread = std::thread(&DMSManager::headposeLoop, this);  // Start headpose detection in its own thread
	    //eyegazeThread = std::thread(&DMSManager::eyegazeLoop, this);  // Start eyegaze detection in its own thread
	    tcpThread = std::thread(&DMSManager::commtcpLoop, this); // Start tcp thread in its own thread
	    //vehicleStateThread = std::thread(&DMSManager::vehicleStateLoop, this); // Start vehicle state in its own thread
	    //postProcessingThread = std::thread(&DMSManager::postprocessingLoop, this); // Start post processing in its own thread
	    commandsThread = std::thread(&DMSManager::commandsLoop, this); // Start commands thread in its own thread
	    //faultsThread = std::thread(&DMSManager::faultsLoop, this); // Start faults thread in its own thread
	    firstRun = false ;    
            return true;
	}
	else{
	    //starting each component in its own thread
	    cameraComponent.initialize("/dev/video0");
	    cameraThread = std::thread(&DMSManager::cameraLoop, this);  // Start the camera loop in its own thread
	    //preprocessingThread = std::thread(&DMSManager::preprocessingLoop, this);  // Start the preprocessing loop in its own thread
	    //faceDetectionComponent.initialize("/home/dms/DMS/ModularCode/modelconfigs/yoloface-500k-v2.cfg", "/home/dms/DMS/ModularCode/ modelconfigs/yoloface-500k-v2.weights");
	    faceDetectionThread = std::thread(&DMSManager::faceDetectionLoop, this);  // Start face detection in its own thread
	    //drowsinessThread = std::thread(&DMSManager::drowsinessLoop, this);  // Start drowsiness detection in its own thread
	    headposeThread = std::thread(&DMSManager::headposeLoop, this);  // Start headpose detection in its own thread
	    //eyegazeThread = std::thread(&DMSManager::eyegazeLoop, this);  // Start eyegaze detection in its own thread
	    //tcpThread = std::thread(&DMSManager::commtcpLoop, this); // Start tcp thread in its own thread
	    //vehicleStateThread = std::thread(&DMSManager::vehicleStateLoop, this); // Start vehicle state in its own thread
	    //postProcessingThread = std::thread(&DMSManager::postprocessingLoop, this); // Start post processing in its own thread
	    //commandsThread = std::thread(&DMSManager::commandsLoop, this); // Start commands thread in its own thread
	    //faultsThread = std::thread(&DMSManager::faultsLoop, this); // Start faults thread in its own thread
	    //----------------------------------------
	    cameraComponent.startCapture();
	    //preprocessingComponent.startProcessing();
	    faceDetectionComponent.startDetection();
	    //drowsinessComponent.startDrowsinessDetection();
	    headposeComponent.startHeadPoseDetection();
	    //eyegazeComponent.startEyeGazeDetection();
	    //tcpComponent.startServer();
	    //vehicleStateManager.startStateManager();
	    //postProcessingComponent.postProcess();
	    //faultManager.faultstart();
		firstRun = false ;
		return true;
	}
}



void DMSManager::stopSystem() {

    running = false;  // Signal all loops to stop
     
    clearQueues();
    // Log performance metrics
    headposeComponent.logPerformanceMetrics();
    tcpComponent.logDataTransferMetrics();
    faceDetectionComponent.logPerformanceMetrics();

    cameraComponent.stopCapture();
    preprocessingComponent.stopProcessing();
    faceDetectionComponent.stopDetection();
    drowsinessComponent.stopDrowsinessDetection();
    headposeComponent.stopHeadPoseDetection();
    eyegazeComponent.stopEyeGazeDetection();
    //tcpComponent.stopServer();
    //vehicleStateManager.stopStateManager();
    //postProcessingComponent.stopPostProcess();
    //faultManager.faultstop();
	

   
    // Wait for each thread to finish its task and join
    if (cameraThread.joinable()) cameraThread.join();
    //if (preprocessingThread.joinable()) preprocessingThread.join();
    if (faceDetectionThread.joinable()) faceDetectionThread.join();
    //if (drowsinessThread.joinable()) drowsinessThread.join();
    if (headposeThread.joinable()) headposeThread.join();
    //if (eyegazeThread.joinable()) eyegazeThread.join();
    //if (tcpThread.joinable()) tcpThread.join();
    //if (vehicleStateThread.joinable()) vehicleStateThread.join();
    //if (postProcessingThread.joinable()) postProcessingThread.join();
    //if (commandsThread.joinable()) commandsThread.join();
    //if (faultsThread.joinable()) faultsThread.join();


    // ADD Additional cleanup if necessary
    //----------------------------------------------------------------------------------------------------------------
}



void DMSManager::setupSignalHandlers() {
    signal(SIGPIPE, SIG_IGN);  // Ignore SIGPIPE
}

//component loops that start in their own thread
void DMSManager::cameraLoop() {
    cameraComponent.startCapture();
}
void DMSManager::preprocessingLoop() {
    preprocessingComponent.startProcessing();
}
void DMSManager::faceDetectionLoop() {
    faceDetectionComponent.startDetection();
}
void DMSManager::drowsinessLoop() {
    drowsinessComponent.startDrowsinessDetection();
}
void DMSManager::headposeLoop() {
    headposeComponent.startHeadPoseDetection();
}
void DMSManager::eyegazeLoop() {
    eyegazeComponent.startEyeGazeDetection();
}
void DMSManager::commtcpLoop() {
    setupSignalHandlers();
    tcpComponent.startServer();
}
void DMSManager::vehicleStateLoop() {
    vehicleStateManager.startStateManager();
}
void DMSManager::postprocessingLoop(){
    postProcessingComponent.postProcess();
}

void DMSManager::faultsLoop(){
    faultManager.faultstart();
}

//loop for DMSmanager component to check for any needed commands by components
void DMSManager::commandsLoop(){
    std::string command;
    while (true) {//changed to true
        if (commandsQueue.tryPop(command)) {
            std::cout << "Received command in the dms manager "<< command << std::endl;
	        this->handlecommand(command);
        }
    }
}



//function for different congurations-----------------------------------------------------------------------


//setting fps for camera
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
    preprocessingQueue.clear();
    faceDetectionQueue.clear();
    faceRectQueue.clear();
    drowsinessDetectionQueue.clear();
    headposeDetectionQueue.clear();
    eyegazeDetectionQueue.clear();
    framesQueue.clear();
    eyegazeframesQueue.clear();
    tcpOutputQueue.clear();
    stateOutputQueue.clear();
    postOutputQueue.clear();
}



void DMSManager::handlecommand(std::string& command) {
	clearQueues();
	// Example model paths
	std::map<std::string, std::string> headPoseModels = {
		{"AX", "/home/dms/DMS/ModularCode/include/Ax.engine"},
		{"AY", "/home/dms/DMS/ModularCode/include/Ay.engine"},
		{"AZ", "/home/dms/DMS/ModularCode/include/Az.engine"},
		{"A0", "/home/dms/DMS/ModularCode/include/A0.engine"},
		{"eff0", "/home/dms/DMS/ModularCode/include/eff0.engine"},
		{"eff1", "/home/dms/DMS/ModularCode/include/eff1.engine"},
		{"eff2", "/home/dms/DMS/ModularCode/include/eff2.engine"},
		{"eff3", "/home/dms/DMS/ModularCode/include/eff3.engine"},
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
        {"No Face Detection", {"No Face Detection", "No Face Detection"}},
        // Add more models as needed
    };

    //setting FPS
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

    //setting face detection threshold
    else if (command.find("SET_FDT:") != std::string::npos) {

        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string fdtValueStr = command.substr(pos + 1);
            int fdtValue = std::stoi(fdtValueStr);
            std::cout << "Setting Face Detection Threshhold to: " << fdtValue << std::endl;
	    setFaceFDT(fdtValue);	
        } 
	else {
            std::cerr << "Invalid SETFDT command format: " << command << std::endl;
          }
	}
    //setting source
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
    //turning off the system
     else if (command == "TURN_OFF") {
        std::cout << "Turning off..." << std::endl;
	stopSystem();

    //turning on the system
    } else if (command == "TURN_ON") {
        std::cout << "Turning on..." << std::endl;
	startSystem();
    //Clear Queue
    } else if (command == "Clear Queue") {
        clearQueues();
    // Handling Face Detection Model
    }else if (command.find("SET_FD_MODEL:") != std::string::npos) {
        size_t pos = command.find(":");
        if (pos != std::string::npos) {
            std::string modelValue = command.substr(pos + 1);
            std::cout << "Setting Face Detection Model to: " << modelValue << std::endl;
            auto it = faceDetectionModels.find(modelValue);
            if (it != faceDetectionModels.end()) {
                std::string weightPath = it->second.second;
                std::string configPath = it->second.first;
				if (weightPath == "No Face Detection" && configPath == "No Face Detection"){
					faceDetectionComponent.modelstatus = false;
		            std::cout << "Updated Face Detection Model and Config to: " << weightPath << " and " << configPath << std::endl;
		            clearQueues();
				} 
				else{
					faceDetectionComponent.stopDetection();
		            		faceDetectionComponent.initialize(configPath,weightPath);
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

    // Handling Head Pose Model
	}else if (command.find("SET_HP_MODEL:") != std::string::npos) {
		size_t pos = command.find(":");
		if (pos != std::string::npos) {
			std::string modelValue = command.substr(pos + 1);
			std::cout << "Setting Head Pose Model to: " << modelValue << std::endl;
			if (headPoseModels.find(modelValue) != headPoseModels.end()) {
				clearQueues();
			    headposeComponent.updateHeadPoseEngine(headPoseModels[modelValue]);
				clearQueues();
			} else {
			    std::cerr << "Head pose model identifier not recognized: " << modelValue << std::endl;
			}
		} else {
			std::cerr << "Invalid SET_HP_MODEL command format: " << command << std::endl;
		}

    // Handling eye gaze Model
	}else if (command.find("SET_EG_MODEL:") != std::string::npos) {
		size_t pos = command.find(":");
		if (pos != std::string::npos) {
			std::string modelValue = command.substr(pos + 1);
			std::cout << "Setting Eye Gaze Model to: " << modelValue << std::endl;
			if (eyeGazeModels.find(modelValue) != eyeGazeModels.end()) {
				clearQueues();
			    headposeComponent.updateEyeGazeEngine(eyeGazeModels[modelValue]);
				clearQueues();
			} else {
			    std::cerr << "Eye gaze model identifier not recognized: " << modelValue << std::endl;
			}
		} else {
		    std::cerr << "Invalid SET_EG_MODEL command format: " << command << std::endl;
			}

   // Handle unknown command
    } else { 
        std::cerr << "Unknown command: " << command << std::endl;
    }
}





//Initialization functions that are needed for some components-----------------------------------------------
bool DMSManager::initializeCamera(const std::string& source) {
    return cameraComponent.initialize(source);
}
bool DMSManager::initializeFaceDetection(const std::string& modelConfiguration, const std::string& modelWeights) {
    //return faceDetectionComponent.initialize(modelConfiguration, modelWeights);
    return true;
}

//*******if needed for drowsiness detection
bool DMSManager::initializeDrowsinessDetection() {
    return drowsinessComponent.initialize();
}
//***//

//*******if needed for headpose detection
bool DMSManager::initializeHeadposeDetection() {
    return headposeComponent.initialize();
}
//***//

//******************missing initialization for the vehicle state to pass the file to read from***************************
//-----------------------------------------------------------------------------------------------------------------------------

