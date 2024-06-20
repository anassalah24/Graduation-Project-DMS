// FaultManager.h

#pragma once

#include <queue>
#include <thread>
#include <iostream>
#include "threadsafequeue.h"

/* Declerations for FPS and FDT */
#define MAX_FPS_THRESHOLD       100
#define MIN_FPS_THRESHOLD       0
#define MAX_FDT_THRESHOLD       100
#define MIN_FDT_THRESHOLD       0

/* Declerations for max velocity and steering */
#define MAX_VELOCITY_THRESHOLD  240
#define MAX_STEERING_THRESHOLD  540

class FaultManager
{
public:
    FaultManager(ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~FaultManager();

    void faultstart();
    void faultstop();
    void faulthandling(const std::string& fault);

private:
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;
    std::thread faultsthread;
    bool running;
    void faultfind();
};


