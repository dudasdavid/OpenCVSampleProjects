# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from piVideoStream import PiVideoStream
import numpy as np
import time
import cv2
from threading import Thread, Lock

frameWidth = 800
frameHeight = 480
frameRate = 20

camera = PiVideoStream(resolution=(frameWidth, frameHeight),framerate=frameRate).start()

# allow the camera to warmup
time.sleep(1)

frame = camera.read()
    
captureFps = 0;
visualizerFps = 0;

mutex = Lock()
stopped = False

timeBegin = time.time()

def cameraCapture():
    global stopped
    global frame
    global mutex
    global captureFps

    frameCounter = 0
    tick = 0
    
    # capture frames from the camera
    while(1):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        mutex.acquire()
        frame = camera.read()
        mutex.release()
     
        frameCounter+=1;
        timeNow = time.time() - timeBegin;
        if (timeNow - tick >= 1):
            tick+=1;
            captureFps = frameCounter;
            frameCounter = 0;
     
        time.sleep(0.01)
        if stopped:
            break
                        
    return
    
    
def visualizer():
    global stopped
    global frame
    global mutex 
    global visualizerFps

    frameCounter = 0
    tick = 0
    
    cv2.namedWindow( "Frame", cv2.WINDOW_NORMAL );
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN);
    
    
    while(1):
        mutex.acquire()
        finalFrame = frame.copy()
        mutex.release()

        cv2.putText(finalFrame, ("Capture FPS=%d" % captureFps ), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0));
        cv2.putText(finalFrame, ("Visualizer FPS=%d" % visualizerFps ), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0));
        
        # show the output frame
        cv2.imshow("Frame", finalFrame)
        key = cv2.waitKey(10) & 0xFF
        
        # if the `ESC` key was pressed, break from the loop
        if key == 27:
            stopped = True
            break
            
        frameCounter+=1;
        timeNow = time.time() - timeBegin;
        if (timeNow - tick >= 1):
            tick+=1;
            visualizerFps = frameCounter;
            frameCounter = 0;
            
    return

    
t1 = Thread(target=cameraCapture, args=())
t2 = Thread(target=visualizer, args=())

t1.start()
t2.start()
    
# loop over the frames from the video stream
while True:
    time.sleep(1)
    if stopped: break

# Clean up
cv2.destroyAllWindows()
camera.stop()