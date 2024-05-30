#include "postprocessingcomponent.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
//---------------------------------------------------------

constexpr int16_t PostProcessingComponent::LOW_SPEED;
constexpr int16_t PostProcessingComponent::MEDIUM_SPEED;
constexpr int16_t PostProcessingComponent::MIN_STEERING_ANGLE;
constexpr int16_t PostProcessingComponent::DIRECTION_STRAIGHT;
constexpr int16_t PostProcessingComponent::DIRECTION_RIGHT;
constexpr int16_t PostProcessingComponent::DIRECTION_LEFT;
constexpr int16_t PostProcessingComponent::HEADPOSE_LEFT;
constexpr int16_t PostProcessingComponent::HEADPOSE_STRAIGHT;
constexpr int16_t PostProcessingComponent::HEADPOSE_RIGHT;
constexpr int16_t PostProcessingComponent::EYEGAZE_FRONT;
constexpr int16_t PostProcessingComponent::EYEGAZE_REARMIRROR;
constexpr int16_t PostProcessingComponent::EYEGAZE_RIGHTMIRROR;
constexpr int16_t PostProcessingComponent::EYEGAZE_LEFTMIRROR;
constexpr int16_t PostProcessingComponent::EYEGAZE_RADIO;
constexpr int16_t PostProcessingComponent::EYEGAZE_DASHBOARD;
constexpr int16_t PostProcessingComponent::SYSTEM_OFF;
constexpr int16_t PostProcessingComponent::ALERT;
constexpr int16_t PostProcessingComponent::NO_ALERT;
constexpr int16_t PostProcessingComponent::DELAY_500MS;
constexpr int16_t PostProcessingComponent::DELAY_1S;
constexpr int16_t PostProcessingComponent::DELAY_1500MS;
constexpr int16_t PostProcessingComponent::DELAY_2S;




//--------------------------------------------------------

// Constructor
PostProcessingComponent::PostProcessingComponent(ThreadSafeQueue<CarState>& inputQueue, ThreadSafeQueue<int>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}

// Destructor
PostProcessingComponent::~PostProcessingComponent() {}

// Helper function to extract a numerical value from a line of text
double PostProcessingComponent::extractValueFromLine(const std::string& line, const std::string& keyword) {
    size_t pos = line.find(keyword);
    if (pos != string::npos) {
        size_t number_pos = line.find_first_of("0123456789", pos + keyword.size());
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





// Method to parse the head pose from a file
void PostProcessingComponent::parseHeadPose(const std::string& headposeFilePath) {
    std::ifstream headposeFile(headposeFilePath);
    if (headposeFile.is_open()) {
        std::string line;
        while (std::getline(headposeFile, line)) {
            headPose.headPoseAngle = extractValueFromLine(line, "Headpose");
            if (headPose.headPoseAngle != 0.0) { // Found the value
                break;
            }
        }
        headposeFile.close();
    } else {
        std::cerr << "Unable to open head pose file: " << headposeFilePath << std::endl;
    }

}

// Method to parse the eye gaze from a file
void PostProcessingComponent::parseEyeGaze(const std::string& eyegazeFilePath) {
    std::ifstream eyegazeFile(eyegazeFilePath);
    if (eyegazeFile.is_open()) {
        std::string line;
        while (std::getline(eyegazeFile, line)) {
            eyeGaze.eyeGazeZone = extractValueFromLine(line, "Eyegaze");
            if (eyeGaze.eyeGazeZone != 0) { // Found the value
                break;
            }
        }
        eyegazeFile.close();
    } else {
        std::cerr << "Unable to open eye gaze file: " << eyegazeFilePath << std::endl;
    }
}


//run thread separately
void PostProcessingComponent::postProcess(){
     if (running) {
        std::cerr << "postProcessing is already running." << std::endl;
        return;
    }
    running = true;
    postThread = std::thread(&PostProcessingComponent::postLoop, this);
}

void PostProcessingComponent::stopPostProcess() {
    running = false;
    if (postThread.joinable()) {
        postThread.join();
    }
}

int16_t PostProcessingComponent::eyeGazeCheck(EyeGaze& eye_gaze, int16_t car_direction)
{
    /* Check which direction the car is moving */
    switch(car_direction)
    {

    /* Car moving straight ahead */
    case DIRECTION_STRAIGHT:

        /* Check eye gaze */
        switch(eye_gaze.eyeGazeZone)
        {
        case EYEGAZE_FRONT:         return NO_ALERT;
            break;
        case EYEGAZE_REARMIRROR :
        case EYEGAZE_RIGHTMIRROR:
        case EYEGAZE_LEFTMIRROR :
        case EYEGAZE_RADIO      :
        case EYEGAZE_DASHBOARD  :

            /* Check again after 2 seconds */
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_2S));
            parseEyeGaze(TEXT_FILE_LOCATION);

            if(eye_gaze.eyeGazeZone==EYEGAZE_FRONT)
            {
                return NO_ALERT;
            }
            else
            {
                return ALERT;
            }
            break;

        default:    /* eye gaze not in any secondary zone */

            /* Check again after 0.5 seconds */
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_500MS));
            parseEyeGaze(TEXT_FILE_LOCATION);
            if(eye_gaze.eyeGazeZone==EYEGAZE_FRONT)
            {
                return NO_ALERT;
            }
            else
            {

                return ALERT;
            }
            break;
        }
        break;

    /* Car turning right */
    case DIRECTION_RIGHT:
        switch(eye_gaze.eyeGazeZone)
        {
        case EYEGAZE_RIGHTMIRROR:         return NO_ALERT;
            break;
        case EYEGAZE_FRONT:
        case EYEGAZE_REARMIRROR :

            /* Check again after 1.5 seconds */
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_1500MS));
            parseEyeGaze(TEXT_FILE_LOCATION);
            if(eye_gaze.eyeGazeZone==EYEGAZE_RIGHTMIRROR)
            {
                return NO_ALERT;
            }
            else
            {
                return ALERT;
            }
            break;
        case EYEGAZE_LEFTMIRROR :
        case EYEGAZE_RADIO      :
        case EYEGAZE_DASHBOARD  :

        default:    /* eye gaze not in any secondary zone */
            /* Check again after 0.5 seconds */
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_500MS));
            parseEyeGaze(TEXT_FILE_LOCATION);
            if(eye_gaze.eyeGazeZone==EYEGAZE_RIGHTMIRROR)
            {
                return NO_ALERT;
            }
            else
            {
                return ALERT;
            }
            break;
        }
        break;

    /* Car turning left */
    case EYEGAZE_LEFTMIRROR:
        switch(eye_gaze.eyeGazeZone)
        {
        case EYEGAZE_LEFTMIRROR :         return NO_ALERT;
            break;
        case EYEGAZE_FRONT:
        case EYEGAZE_REARMIRROR :

            /* Check again after 1.5 seconds */
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_1500MS));
            parseEyeGaze(TEXT_FILE_LOCATION);
                        
            if(eye_gaze.eyeGazeZone==EYEGAZE_LEFTMIRROR)
            {
                return NO_ALERT;
            }
            else
            {
                return ALERT;
            }
            break;
        case EYEGAZE_RIGHTMIRROR:
        case EYEGAZE_RADIO      :
        case EYEGAZE_DASHBOARD  :

        default:    /* eye gaze not in any secondary zone */
            /* Check again after 0.5 seconds */
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_500MS));
            parseEyeGaze(TEXT_FILE_LOCATION);
            if(eye_gaze.eyeGazeZone==EYEGAZE_LEFTMIRROR)
            {
                return NO_ALERT;
            }
            else
            {
                return ALERT;
            }
            break;
        }
        break;
    }
    return NO_ALERT;
}

// Function to make decisions based on input data
int16_t PostProcessingComponent::makeDecision(const CarState& state, EyeGaze& eye_gaze, HeadPose& head_pose) {

    // Implementing decision logic based on scenarios described
    if(state.velocity<=LOW_SPEED)
    {
        return SYSTEM_OFF;
    }

    /* velocity more than 15 */
    else
    {
        /* Straight ahead */
        if(state.steeringWheelAngle>(-MIN_STEERING_ANGLE) && state.steeringWheelAngle<MIN_STEERING_ANGLE)
        {
            /* Check head pose */
            if(head_pose.headPoseAngle == HEADPOSE_STRAIGHT)
            {
                return eyeGazeCheck(eye_gaze,DIRECTION_STRAIGHT);
            }
            /* Head pose not straight ahead */
            else
            {
                /* Check again after 0.5 seconds */
                std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_500MS));
                parseHeadPose(TEXT_FILE_LOCATION);
                           
                if(head_pose.headPoseAngle == HEADPOSE_STRAIGHT)
                {
                    return eyeGazeCheck(eye_gaze,DIRECTION_STRAIGHT);
                }
                else
                {
                    return ALERT;
                }
            }
        }
        /* Turning right */
        else if(state.steeringWheelAngle>MIN_STEERING_ANGLE)
        {
            /* Head pose straight ahead or skewed to the turning direction */
            if(head_pose.headPoseAngle==HEADPOSE_STRAIGHT || head_pose.headPoseAngle==HEADPOSE_RIGHT)
            {
                /* Eye gaze to the direction of the mirror corresponding to the rotation */
                return eyeGazeCheck(eye_gaze,DIRECTION_RIGHT);
            }
            else
            {
                /* Check again after 0.5 seconds */
                std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_500MS));
                parseHeadPose(TEXT_FILE_LOCATION);
                if(head_pose.headPoseAngle==HEADPOSE_STRAIGHT || head_pose.headPoseAngle==HEADPOSE_RIGHT)
                {
                    return eyeGazeCheck(eye_gaze,DIRECTION_RIGHT);
                }
                else
                {
                    return ALERT;
                }
            }                
        }
        /* Turning left */
        else if(state.steeringWheelAngle<(-MIN_STEERING_ANGLE))
        {
            /* Head pose straight ahead or skewed to the turning direction */
            if(head_pose.headPoseAngle==HEADPOSE_STRAIGHT || head_pose.headPoseAngle==HEADPOSE_LEFT)
            {
                /* Eye gaze to the direction of the mirror corresponding to the rotation */
                return eyeGazeCheck(eye_gaze,DIRECTION_LEFT);
            }
            else
            {
                /* Check again after 0.5 seconds */
                std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_500MS));
                parseHeadPose(TEXT_FILE_LOCATION);
                if(head_pose.headPoseAngle==HEADPOSE_STRAIGHT || head_pose.headPoseAngle==HEADPOSE_LEFT)
                {
                    return eyeGazeCheck(eye_gaze,DIRECTION_LEFT);
                }
                else
                {
                    return ALERT;
                }
            }
        }
    }

    return NO_ALERT;
}


// loop that gets the headpose eye gaze , car state and sends this information to the make decision function , then sends the decision to the output queue
void PostProcessingComponent::postLoop(){
    CarState carstate;
    const std::string FilePath = TEXT_FILE_LOCATION;

    while (running)
    {

        parseHeadPose(FilePath);
        parseEyeGaze(FilePath);
        if (inputQueue.tryPop(carstate)) {
            std::cout << "Queue from post vehicle state: Steering angle " << carstate.steeringWheelAngle << ", Velocity " << carstate.velocity << std::endl;
            std::cout << "Head angle : " << headPose.headPoseAngle << ", eyegaze : " << eyeGaze.eyeGazeZone << std::endl;
            int ifAlert;
            ifAlert = makeDecision(carstate,eyeGaze,headPose);
            std::cout << "decision making output:"<<ifAlert << std::endl;
            outputQueue.push(ifAlert);


        }
        
    }
    

}
