
#pragma once

#include <string>
#include <thread>
#include <vehiclestatemanager.h>
#include "threadsafequeue.h"


// Define structures for input data 
struct HeadPose {
    int16_t headPoseAngle;
};

struct EyeGaze {
    int16_t eyeGazeZone;
};

class PostProcessingComponent {
public:
    PostProcessingComponent(ThreadSafeQueue<CarState>& inputQueue, ThreadSafeQueue<int>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~PostProcessingComponent();

    void parseHeadPose(const std::string& headposeFilePath);
    void parseEyeGaze(const std::string& eyegazeFilePath);

    void postProcess();
    void stopPostProcess();

    int16_t eyeGazeCheck(EyeGaze& eye_gaze, int16_t car_direction);
    int16_t makeDecision(const CarState& state, EyeGaze& eye_gaze, HeadPose& head_pose);


    double extractValueFromLine(const std::string& line, const std::string& keyword);
    static constexpr int16_t LOW_SPEED = 10;
    static constexpr int16_t MEDIUM_SPEED = 40;
    static constexpr int16_t MIN_STEERING_ANGLE = 30;
    static constexpr int16_t DIRECTION_STRAIGHT = 1;
    static constexpr int16_t DIRECTION_RIGHT = 2;
    static constexpr int16_t DIRECTION_LEFT = 3;
    static constexpr int16_t HEADPOSE_LEFT = 1;
    static constexpr int16_t HEADPOSE_STRAIGHT = 2;
    static constexpr int16_t HEADPOSE_RIGHT = 3;
    static constexpr int16_t EYEGAZE_FRONT = 1;
    static constexpr int16_t EYEGAZE_REARMIRROR = 2;
    static constexpr int16_t EYEGAZE_RIGHTMIRROR = 3;
    static constexpr int16_t EYEGAZE_LEFTMIRROR = 4;
    static constexpr int16_t EYEGAZE_RADIO = 5;
    static constexpr int16_t EYEGAZE_DASHBOARD = 6;
    static constexpr int16_t SYSTEM_OFF = 0;
    static constexpr int16_t ALERT = 1;
    static constexpr int16_t NO_ALERT = 2;
    static constexpr int16_t DELAY_500MS = 500;
    static constexpr int16_t DELAY_1S = 1000;
    static constexpr int16_t DELAY_1500MS = 1500;
    static constexpr int16_t DELAY_2S = 2000;


private:
    ThreadSafeQueue<CarState>& inputQueue; 
    ThreadSafeQueue<int>& outputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;

    HeadPose headPose;
    EyeGaze eyeGaze;
    std::thread postThread;
    bool running;
    void postLoop();



};

