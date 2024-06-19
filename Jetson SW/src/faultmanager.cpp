// FaultManager.cpp

#include "faultmanager.h"


//constructor
FaultManager::FaultManager(ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue) 
: commandsQueue(commandsQueue), faultsQueue(faultsQueue), running(false){}

//destructor
FaultManager::~FaultManager() {
    faultstop();
}

//starting the fault manager thread
void FaultManager::faultstart()
{
    if (running) {
        std::cerr << "Fault manager is already running." << std::endl;
        return;
    }
    running = true;
    faultsthread = std::thread(&FaultManager::faultfind, this);
}


//function to release the thread ( add any cleanup needed )
void FaultManager::faultstop() {
    running = false;
    if (faultsthread.joinable()) {
        faultsthread.join();
    }
}


//function to poll on queue to check for any faults sent
void FaultManager::faultfind()
{
    std::string fault;
    while (running)
    {
        if (faultsQueue.tryPop(fault))
        {
            std::cout << "Received fault in the fault manager "<< fault << std::endl;
            this->faulthandling(fault);
        }
    }
}


//function to identify the fault sent
void FaultManager::faulthandling(const std::string &fault)
{

    std::string command;

    /********************** Camera component fault *******************/

    //Camera can't connect
    if (fault == "Camera_disconnected")
    {
        command = "Read_video";         //Read from video file
        std::cout<<"Live feed not found, connecting to video"<<std::endl;
        commandsQueue.push(command);
    }


    /************************* Commtcp faults ****************************/
    //In commtcp.cpp
    /*
    fpsvalue = std::stoi(message.substr(8));
    if(fpsvalue<MIN_FPS_THRESHOLD || fpsvalue>MAX_FPS_THRESHOLD)
    {
        faultsQueue.push(command); //with the same command that should've been sent to DMS
    }
    //same for FDT
    */

    //FPS more than max threshold or less than min threshold
    else if (fault.find("SET_FPS") != std::string::npos)
    {
        std::cout << "Incorrect FPS sent" << std::endl;
        size_t pos = fault.find(":");
        if (pos != std::string::npos)
        {
            std::string fpsValueStr = fault.substr(pos + 1);
            int fpsvalue = std::stoi(fpsValueStr);
            if(fpsvalue<MIN_FPS_THRESHOLD)
            {
                fpsvalue = MIN_FPS_THRESHOLD;
                command = "SET_FPS:" + std::to_string(fpsvalue);
                //Send new FPS value to DMS manager
                commandsQueue.push(command);
            }
            else if(fpsvalue>MAX_FPS_THRESHOLD)
            {
                fpsvalue = MAX_FPS_THRESHOLD;
                command = "SET_FPS:" + std::to_string(fpsvalue);
                //Send new FPS value to DMS manager
                commandsQueue.push(command);
            }
            else //No fault in the fps
            {
                command = "SET_FPS:" + fault.substr(8);
                //Send same FPS to DMS manager
                commandsQueue.push(command);
            }
        }
    }

    //FDT more than max threshold or less than min threshold
    else if (fault.find("SET_FDT:") != std::string::npos)
    {
        std::cout << "Incorrect FDT sent" << std::endl;
        size_t pos = fault.find(":");
        if (pos != std::string::npos)
        {
            std::string fdtValueStr = fault.substr(pos + 1);
            int fdtvalue = std::stoi(fdtValueStr);
            if(fdtvalue<MIN_FDT_THRESHOLD)
            {
                fdtvalue = MIN_FDT_THRESHOLD;
                command = "SET_FDT:" + std::to_string(fdtvalue);
                //Send new FDT value to DMS manager
                commandsQueue.push(command);
            }
            else if(fdtvalue>MAX_FDT_THRESHOLD)
            {
                fdtvalue = MAX_FDT_THRESHOLD;
                command = "SET_FDT:" + std::to_string(fdtvalue);
                //Send new FDT value to DMS manager
                commandsQueue.push(command);
            }
            else //No fault in the fdt
            {
                command = "SET_FDT:" + fault.substr(8);
                //Send same FDT to DMS Manager
                commandsQueue.push(command);
            }
        }
    }

    /* System turn off request at high vehicle velocity */
    //The request will come from TCP to fault
    //In DMS
    /*
    else if (command == "TURN_OFF")
    {
        VehicleStateManager vehicleManager;
        CarState carState = vehicleManager.getCarState();
        //Get velocity value stored in the struct
        double velocity = carState.velocity;
        if(velocity>40)
        {
            //Send back to user
            // Velocity high, can't turn off system at the moment
            // Please slow down to turn off
            //
        }
        else
        {
            std::cout << "Turning off..." << std::endl;
            stopSystem();
        }
    }
    */
    else if(fault == "TURN_OFF")
    {
        command = "TURN_OFF";
        commandsQueue.push(command);
    }

    /*
     * No connection to the other device established
     * So DMS can close other components till connection is restored
     */
    else if(fault == "TCP_Connection_Error")
    {
        command = "No_TCP_Connection";
        commandsQueue.push(command);
    }

    /************************** Face detection Faults ********************************/

    //if weights file not found
    else if (fault == "PrePrc_fault")
    {
        std::cout << "Weights file not found" << std::endl;
        command = "Close_PrePro";
        commandsQueue.push(command);
    }

    /**************************** Vehicle state manager fault *************************/
    //Velocity in vehicle state fault
    //In vehicle state manager
    /*
    extractedValue = extractValueFromLine(line, "velocity");
    if(extractedValue>MAX_VELOCITY_THRESHOLD)
    {
        std::string fault="Velocity_fault";
        faultsQueue.push(fault);
    }
    //same with steering
    */
    else if (fault.find("Velocity_fault") != std::string::npos)
    {
        std::cout << "Velocity above max threshold, setting velocity to max" << std::endl;
        int velocity = MAX_VELOCITY_THRESHOLD;
        command = "Velocity_max:" + std::to_string(velocity);
        commandsQueue.push(command);
    }
    else if (fault.find("Steering_fault") != std::string::npos)
    {
        std::cout << "Steering above max threshold, setting steering to max" << std::endl;
        int steering = MAX_STEERING_THRESHOLD;
        command = "Steering_max:" + std::to_string(steering);
        commandsQueue.push(command);
    }

    // Handle unknown fault
    else
    {
        std::cerr << "Unknown fault: " << fault << std::endl;
    }
}
