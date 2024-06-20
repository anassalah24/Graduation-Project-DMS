# Jetson SW Application

## Overview
This application runs on a Jetson Nano board and is responsible for real-time detection of the driver's face , head pose and eye gaze.
## Features
- **Real-Time Face Detection**: Detects driver's head.
- **Real-Time Head Pose Detection**: Detects the orientation of the driver's head.
- **Real-Time Eye Gaze Detection**: Monitors the driver's eye gaze direction.
- **Modular Architecture**: Easily extendable to add new functionalities.
- **TCP Communication**: Communicates detected data to the Windows application via TCP.
- **Multithreaded**: utilizes multithreading and multithreading control to achieve a real time experience.

## Installation (incase you have a new board)
### Prerequisites
- Jetson Nano with JetPack SDK installed (4.6.4)
- Required libraries:
![image](https://github.com/anassalah24/Graduation-Project-DMS/assets/68183749/99495e6c-363f-4e51-99cb-dd7b5041624a)

### Setting Up the Environment
1. Clone the repository:
    ```bash
    git clone https://github.com/anassalah24/Graduation-Project-DMS.git
    cd Graduation-Project-DMS/Jetson SW
    ```

2. Set up the environment by running the setup script:
    ```bash
    setup_environment.sh
    ```

4. Set up the environment as shown in the next section:
   - Make a directory inside `/home` and name it "DMS":
     ```bash
     mkdir -p /home/DMS
     ```

   - Create two folders inside "DMS", name one "ModularCode" and the other "Videos":
     ```bash
     mkdir -p /home/DMS/ModularCode
     mkdir -p /home/DMS/Videos
     ```
![image](https://github.com/anassalah24/Graduation-Project-DMS/assets/68183749/a6482c68-b49a-4233-9aee-af5dfef9cfd0)

   - Copy and paste all files inside `/Jetson SW` to `/home/DMS/ModularCode`:
     ```bash
     cp -r * /home/DMS/ModularCode
     ```
   - Get a copy of the folder "Modelsconfigs" from the owner of the repository and place it inside `/home/DMS/ModularCode`.
![image](https://github.com/anassalah24/Graduation-Project-DMS/assets/68183749/f8ebac9b-d75a-4934-a411-f7e5455af8e8)

   - Get a copy of the models engine files from the owner of the repository and place it inside `/home/DMS/ModularCode/include`
![image](https://github.com/anassalah24/Graduation-Project-DMS/assets/68183749/ce3867cd-4807-4dbb-b5e2-eb462f91533d)

   - Open a terminal inside `/home/DMS/ModularCode` and write the following commands:
     ```bash
     make clean  
     make
     ./bin/myApplication
     ```
   - connect to the board from the desktopapp using the board's ip (depending on the network) and port(12345) 
### Incase You have the original jetson nano board
  - simply bootup the board
  - press the button on the breadboard connected to the jetson board
  - the app will start automatically
  - connect to the board from the desktopapp using the board's ip (depending on the network) and port(12345) 


## Usage
- Ensure the Jetson Nano is connected to the same network as the Windows application.
