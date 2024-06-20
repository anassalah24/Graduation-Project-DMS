#!/bin/bash

# Update and upgrade the system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libopencv-dev libboost-all-dev libbenchmark-dev
sudo apt-get install -y python3 python3-pip

# Install TensorRT
sudo apt-get install -y nvinfer-runtime-trt-repo-<distribution>-<version>
sudo apt-get install -y libnvinfer8 libnvinfer-dev

# Set up environment variables
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create necessary directories
mkdir -p include src obj bin benchmark

# Compile the application
make

# Instructions for the user
echo "Environment setup complete."
echo "To run the application, use the following command:"
echo "LD_LIBRARY_PATH=/usr/local/lib ./bin/myApplication"
