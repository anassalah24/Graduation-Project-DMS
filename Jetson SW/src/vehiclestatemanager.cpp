#include "vehiclestatemanager.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

//constructor
VehicleStateManager::VehicleStateManager(ThreadSafeQueue<CarState>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
:outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue),running(false) {}

//destructor
VehicleStateManager::~VehicleStateManager() {}


//starting the component
void VehicleStateManager::startStateManager() {
    if (running) {
        std::cerr << "state is already running." << std::endl;
        return;
    }
    running = true;
    stateThread = std::thread(&VehicleStateManager::stateLoop, this);
}

void VehicleStateManager::stopStateManager() {
    running = false;
    if (stateThread.joinable()) {
        stateThread.join();
    }
}

// Helper function to extract a numerical value from a line of text
double VehicleStateManager::extractValueFromLine(const std::string& line, const std::string& keyword) {
    size_t pos = line.find(keyword);
    if (pos != string::npos) {
        size_t number_pos = line.find_first_of("-0123456789", pos + keyword.size());
        if (number_pos != string::npos) {
            try {
                return stod(line.substr(number_pos));
            } catch (const std::invalid_argument& e) {
                cerr << "Invalid number format in line: " << line << endl;
                return 0.0; // or handle error appropriately
            }
        }
    }
    return 0.0; // Indicates not found or error, depending on your error handling strategy
}


void VehicleStateManager::parseCarState(const std::string& dataFilePath) {
    ifstream dataFile(dataFilePath);
    string line;
    if (dataFile.is_open()) {
        while (getline(dataFile, line)) {

            // Attempt to extract the steering wheel angle and velocity from each line
            double extractedValue = extractValueFromLine(line, "steering");

            if (extractedValue != 0.0) { // Assuming 0.0 means not found or invalid
                state.steeringWheelAngle = extractedValue;

            }

            extractedValue = extractValueFromLine(line, "velocity");
            if (extractedValue != 0.0) { // Assuming 0.0 means not found or invalid
                state.velocity = extractedValue;
            }
        }
        dataFile.close();
    } else {
        cerr << "Unable to open data file: " << dataFilePath << endl;
    }
}

// car state getter function
CarState VehicleStateManager::getCarState() const {
    return state;
}

//main loop
void VehicleStateManager::stateLoop() {
    while (running) {
        // Specify the paths to the files where new data is periodically written
        const std::string FilePath = TEXT_FILE_LOCATION;
        parseCarState(FilePath);
        CarState currentState = getCarState();
        outputQueue.push(currentState);

        // Wait a bit before the next update to avoid flooding with too many updates per second.
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
}

