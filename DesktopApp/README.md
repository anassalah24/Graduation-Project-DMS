# Windows Driver Monitoring Application

## Overview
This application runs on a Windows PC and connects to the Jetson Nano application to view model outputs, frames, and configure the system. It is built using Qt 5.15, providing a user-friendly interface for monitoring and interacting with the driver monitoring system.

![image](https://github.com/anassalah24/Graduation-Project-DMS/assets/68183749/a54351e2-c595-4cde-9c29-27b693e45386)


## Features
- **Live View**: Displays real-time frames from the Jetson Nano.
- **Model Outputs**: Shows the outputs from head pose and eye gaze detection models.
- **System Configuration**: Allows users to configure settings for the Jetson Nano application.
- **Modular Design**: Easy to extend and integrate additional functionalities.
- **Network Communication**: Communicates with the Jetson Nano via TCP.

## Installation

### Prerequisites
- Windows 10/11
- Qt 5.15 installed (including Qt Creator)
- Visual Studio (for compiling the application)
- Required libraries: OpenCV

### Setting Up the Environment
1. Clone the repository:
    ```bash
    git clone https://github.com/anassalah24/Graduation-Project-DMS.git
    cd Graduation-Project-DMS/DesktopApp
    ```

2. Open the project in Qt Creator:
   - Launch Qt Creator.
   - Open the project file (`.pro`) located in the `DesktopApp` directory.

3. Configure the project:
   - Ensure you have the correct kit selected (e.g., MSVC2019 64bit).
   - Configure the project to use the required libraries (OpenCV).

4. Build the project:
   - Click on the "Build" button in Qt Creator.
   - Ensure there are no build errors and the application compiles successfully.

### Running the Application
1. Ensure the Jetson Nano application is running and connected to the same network as the Windows PC.
2. Launch the compiled application from Qt Creator or directly from the build directory.
3. Enter IP(depending on the network) and port (12345) of the jetson board to connect to it and view live feed and start configuring the system remotely.
   



