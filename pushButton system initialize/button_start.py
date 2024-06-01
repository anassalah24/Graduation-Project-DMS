import RPi.GPIO as GPIO
import os
import time
import logging
import subprocess

logging.basicConfig(filename='/home/dms/button_start.log', level=logging.DEBUG)

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
inPin = 15
GPIO.setup(inPin, GPIO.IN)  # Removed pull_up_down parameter since Jetson.GPIO does not support it

# Directory to change to and command to run
directory = "/home/dms/DMS/ModularCode/bin"
command = "./myApplication > /home/dms/myApplication.log 2>&1"

# Global variable to keep track of the process
app_process = None

def is_application_running():
    result = subprocess.Popen(['pgrep', '-f', 'myApplication'], stdout=subprocess.PIPE)
    output, _ = result.communicate()
    return result.returncode == 0

def start_application():
    global app_process
    logging.debug('Starting application...')
    app_process = subprocess.Popen(['bash', '-c', 'cd {} && {}'.format(directory, command)])
    logging.debug('Application started.')

def stop_application():
    logging.debug('Stopping application...')
    os.system('pkill -f myApplication')
    logging.debug('Application stopped.')

def toggle_application():
    if is_application_running():
        stop_application()
    else:
        start_application()

try:
    while True:
        button_state = GPIO.input(inPin)
        if button_state == GPIO.LOW:  # Button is pressed
            logging.debug('Button pressed, toggling application...')
            toggle_application()
            time.sleep(1)  # Debounce delay to prevent multiple triggers
finally:
    GPIO.cleanup()
    logging.debug('GPIO cleanup completed.')

