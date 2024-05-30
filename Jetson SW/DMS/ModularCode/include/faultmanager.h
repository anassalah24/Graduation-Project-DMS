// FaultManager.h

#pragma once

#include <queue>
#include <thread>
#include <iostream>
#include "threadsafequeue.h"

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


