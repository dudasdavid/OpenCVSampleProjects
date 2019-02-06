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

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS["mosse"]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

camera = PiVideoStream(resolution=(frameWidth, frameHeight),framerate=frameRate).start()

# allow the camera to warmup
time.sleep(1)

frame = camera.read()
    
captureFps = 0;
visualizerFps = 0;
trackFps = 0

trackerShadow = [False,None]

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
    
def objectTracker():
    global stopped
    global trackFps
    global frame
    global trackerShadow
    global mutex
    
    frameCounter = 0
    tick = 0
    x,y,w,h = 0,0,0,0
    
    success = False
    
    while(1):
        mutex.acquire()
        trackFrame = frame.copy()
        mutex.release()
        
        success = False
    
        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(trackFrame)
     
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
     
        mutex.acquire()
        trackerShadow = [success, (x, y, w, h)]
        mutex.release()
        
        frameCounter+=1;
        timeNow = time.time() - timeBegin;
        if (timeNow - tick >= 1):
            tick+=1;
            trackFps = frameCounter;
            frameCounter = 0;

        if stopped:
            break
    return
    
def visualizer():
    global stopped
    global frame
    global mutex 
    global visualizerFps
    global trackFps
    global captureFps
    global initBB
    global tracker
    global trackerShadow

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
        cv2.putText(finalFrame, ("Tracking FPS=%d" % trackFps ), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0));
        
        mutex.acquire()
        if (trackerShadow[0] != False):
            x = trackerShadow[1][0]
            y = trackerShadow[1][1]
            w = trackerShadow[1][2]
            h = trackerShadow[1][3]
            cv2.rectangle(finalFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        mutex.release()

        # show the output frame
        cv2.imshow("Frame", finalFrame)
        key = cv2.waitKey(10) & 0xFF
        
        # if the `ESC` key was pressed, break from the loop
        if key == 27:
            stopped = True
            break
            
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", finalFrame, fromCenter=False,
                showCrosshair=True)
     
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker = OPENCV_OBJECT_TRACKERS["mosse"]() 
            tracker.init(finalFrame, initBB)
            
        frameCounter+=1;
        timeNow = time.time() - timeBegin;
        if (timeNow - tick >= 1):
            tick+=1;
            visualizerFps = frameCounter;
            frameCounter = 0;
            
    return

    
t1 = Thread(target=cameraCapture, args=())
t2 = Thread(target=visualizer, args=())
t3 = Thread(target=objectTracker, args=())

t1.start()
t2.start()
t3.start()
    
# loop over the frames from the video stream
while True:
    time.sleep(1)
    if stopped: break

# Clean up
cv2.destroyAllWindows()
camera.stop()