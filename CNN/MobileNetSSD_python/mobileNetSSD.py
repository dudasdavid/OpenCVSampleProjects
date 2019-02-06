# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from piVideoStream import PiVideoStream
import numpy as np
import imutils
import time
import cv2
from mvnc import mvncapi
from threading import Thread, Lock

piCamera = True
frameWidth = 800
frameHeight = 480
frameRate = 20
confidenceThreshold = 0.4

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Set logging level and initialize/open the first NCS we find
# mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, mvncapi.LogLevel.DEBUG)

# Initialize and open a device
print("[INFO] initializing device...")
devices = mvncapi.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    exit()
device = mvncapi.Device(devices[0])
device.open()

# Initialize a graph from file at some GRAPH_FILEPATH
print("[INFO] loading model...")
with open('../NeuralNetworks/MobileNetSSD/MobileNetSSD_graph', mode='rb') as f:
    graphBuffer = f.read()
graph = mvncapi.Graph('graph1')

# Allocate the graph to the device
graph.allocate(device, graphBuffer)

# Get the graphTensorDescriptor structs (they describe expected graph input/output)
inputDescriptors = graph.get_option(mvncapi.GraphOption.RO_INPUT_TENSOR_DESCRIPTORS)
outputDescriptors = graph.get_option(mvncapi.GraphOption.RO_OUTPUT_TENSOR_DESCRIPTORS)

# Create input/output Fifos
inputFifo = mvncapi.Fifo('input1', mvncapi.FifoType.HOST_WO)
outputFifo = mvncapi.Fifo('output1', mvncapi.FifoType.HOST_RO)

# Set the Fifo data type to FP32
inputFifo.set_option(mvncapi.FifoOption.RW_DATA_TYPE, mvncapi.FifoDataType.FP16)
outputFifo.set_option(mvncapi.FifoOption.RW_DATA_TYPE, mvncapi.FifoDataType.FP16)

inputFifo.allocate(device, inputDescriptors[0], 2)
outputFifo.allocate(device, outputDescriptors[0], 2)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
if piCamera:
    camera = PiVideoStream(resolution=(frameWidth, frameHeight),framerate=frameRate).start()
else:
    camera = cv2.VideoCapture(0)

    # camera.set(cv2.CAP_PROP_FOURCC, codec)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH ,frameWidth);
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT ,frameHeight);
    camera.set(cv2.CAP_PROP_FPS, frameRate)

time.sleep(1)

timeBegin = time.time()
if piCamera:
    frame = camera.read()
else:
    ret_val, frame = camera.read()
    
detections = []
detectionsShadow = []

captureFps = 0;
visualizerFps = 0;
nnFps = 0;

mutex = Lock()
stopped = False

label_background_color = (125, 175, 75)
label_text_color = (255, 255, 255)  # white text

# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - (127.5, 127.5, 127.5)
    img = img * 0.007843
    return img

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
    
    
def neuralNetwork():
    global stopped
    global nnFps
    global frame
    global detections
    global detectionsShadow
    global mutex
    
    frameCounter = 0
    tick = 0
    
    while(1):
        mutex.acquire()
        detectFrame = frame.copy()
        mutex.release()
        (h, w) = detectFrame.shape[:2]
        
        inputTensor = preprocess_image(detectFrame)
        # cv2.imshow("smallframe", inputTensor)
        
        # Write the image to the input queue
        inputFifo.write_elem(inputTensor.astype(np.float16), None)
        
        # Queue the inference
        graph.queue_inference(inputFifo, outputFifo)
        
        # Get the results from the output queue
        output, userObj = outputFifo.read_elem()
        #   a.	First fp16 value holds the number of valid detections = num_valid.
        #   b.	The next 6 values are unused.
        #   c.	The next (7 * num_valid) values contain the valid detections data
        #       Each group of 7 values will describe an object/box These 7 values in order.
        #       The values are:
        #         0: image_id (always 0)
        #         1: class_id (this is an index into labels)
        #         2: score (this is the probability for the class)
        #         3: box left location within image as number between 0.0 and 1.0
        #         4: box top location within image as number between 0.0 and 1.0
        #         5: box right location within image as number between 0.0 and 1.0
        #         6: box bottom location within image as number between 0.0 and 1.0


        # number of boxes returned
        num_valid_boxes = int(output[0])
        # print('total num boxes: ' + str(num_valid_boxes))

        detections = []
        
        for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not np.isfinite(output[base_index]) or
                    not np.isfinite(output[base_index + 1]) or
                    not np.isfinite(output[base_index + 2]) or
                    not np.isfinite(output[base_index + 3]) or
                    not np.isfinite(output[base_index + 4]) or
                    not np.isfinite(output[base_index + 5]) or
                    not np.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                # print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            x1 = int(output[base_index + 3] * w)
            y1 = int(output[base_index + 4] * h)
            x2 = int(output[base_index + 5] * w)
            y2 = int(output[base_index + 6] * h)

            if (x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0):
                continue

            if (x1 > w or y1 > h or x2 > w or y2 > h ):
                continue
            
            confidence = output[base_index + 2]
            idx = int(output[base_index + 1])

            label_text = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = x1
            label_top = y1 - label_size[1]
            if (label_top < 1):
                label_top = 1
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            
            detections.append([idx, confidence, x1, y1, x2, y2, label_text, label_left, label_top, label_right, label_bottom ])
        
        mutex.acquire()
        detectionsShadow = detections[:]
        mutex.release()
        
        frameCounter+=1;
        timeNow = time.time() - timeBegin;
        if (timeNow - tick >= 1):
            tick+=1;
            nnFps = frameCounter;
            frameCounter = 0;

        if stopped:
            break
    return
    
def visualizer():
    global stopped
    global frame
    global mutex 
    global visualizerFps
    global nnFps
    global captureFps
    global detectionsShadow

    frameCounter = 0
    tick = 0
    
    cv2.namedWindow( "Frame", cv2.WINDOW_NORMAL );
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN);
    
    
    while(1):
        mutex.acquire()
        finalFrame = frame.copy()
        mutex.release()

        mutex.acquire()
        for i in range(0,len(detectionsShadow)):
            confidence = detectionsShadow[i][1]
            if confidence > confidenceThreshold:
                idx = detectionsShadow[i][0]
                x1  = detectionsShadow[i][2]
                y1  = detectionsShadow[i][3]
                x2  = detectionsShadow[i][4]
                y2  = detectionsShadow[i][5]
                label_text   = detectionsShadow[i][6]
                label_left   = detectionsShadow[i][7]
                label_top    = detectionsShadow[i][8]
                label_right  = detectionsShadow[i][9]
                label_bottom = detectionsShadow[i][10]
                
                cv2.rectangle(finalFrame, (x1, y1), (x2, y2), COLORS[idx], 2)
                cv2.rectangle(finalFrame, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                              label_background_color, -1)
                cv2.putText(finalFrame, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
        mutex.release()
        
        cv2.putText(finalFrame, ("Capture FPS=%d" % captureFps ), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0));
        cv2.putText(finalFrame, ("Visualizer FPS=%d" % visualizerFps ), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0));
        cv2.putText(finalFrame, ("NN FPS=%d" % nnFps ), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0));
        
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
t2 = Thread(target=neuralNetwork, args=())
t3 = Thread(target=visualizer, args=())

t1.start()
t2.start()
t3.start()
    
# loop over the frames from the video stream
while True:
    time.sleep(1)
    if stopped: break

# Clean up
inputFifo.destroy()
outputFifo.destroy()
graph.destroy()
device.close()
device.destroy()
cv2.destroyAllWindows()
camera.stop()