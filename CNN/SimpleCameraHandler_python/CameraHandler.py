# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
from threading import Thread, Lock

frameWidth = 800
frameHeight = 480
frameRate = 20


camera = PiCamera()
camera.resolution = (frameWidth, frameHeight)
camera.framerate = frameRate
camera.rotation = 270
rawCapture = PiRGBArray(camera, size=(frameWidth, frameHeight))
stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
# allow the camera to warmup
time.sleep(0.1)

for f in stream:
    frame = f.array
    rawCapture.truncate(0)
    break
    
captureFps = 0;
visualizerFps = 0;

mutex = Lock()
stopped = False

timeBegin = time.time()

def cameraCapture():
    global stopped
    global frame
    global mutex
    global stream
    global rawCapture
    global captureFps

    frameCounter = 0
    tick = 0
    
    # capture frames from the camera
    for f in stream:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        mutex.acquire()
        frame = f.array
        mutex.release()

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
     
        frameCounter+=1;
        timeNow = time.time() - timeBegin;
        if (timeNow - tick >= 1):
            tick+=1;
            captureFps = frameCounter;
            frameCounter = 0;
     
        # time.sleep(0.02)
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
stream.close()
rawCapture.close()
camera.close()