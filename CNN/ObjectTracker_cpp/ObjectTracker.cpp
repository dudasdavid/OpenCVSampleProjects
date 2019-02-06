#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 

#include <ctime>
#include <cmath>
#include <time.h>

#include <pthread.h>
#include <wiringPi.h> 
#include <wiringSerial.h> 

#include "opencv2/videoio.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <opencv2/tracking.hpp>

#include <raspicam/raspicam_cv.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <mvnc.h>

// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100

using namespace std;
using namespace cv;
using namespace cv::dnn;

int faceDetectionEna = 1;
int objectTrackingEna = 1;
int withScreen = 1;
int useMovidius = 1;
int withUART = 0;
int armMovementEna = 0;

pthread_mutex_t lock;

int frameWidth = 800;
int frameHeight = 480;
int cameraIndex = 0;
int netSize = 300;
float confidenceThreshold = 0.4;

String modelTxt = "../NeuralNetworks/MobileNetSSD/MobileNetSSD_deploy.prototxt";
String modelBin = "../NeuralNetworks/MobileNetSSD/MobileNetSSD_deploy.caffemodel";
const char* graphFileName = "../NeuralNetworks/MobileNetSSD/MobileNetSSD_graph";

// List of tracker types in OpenCV 3.4.1
string trackerTypes[7] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE"};
// Create a tracker
string trackerType = trackerTypes[6];
Ptr<Tracker> tracker;
cv::Rect2d inputROI;
cv::Rect2d outBox;
cv::Rect2d outBoxShadow;
bool successShadow = false;
int ROIset = 0;

cv::Mat frame(frameHeight,frameWidth,CV_8UC3,Scalar::all(0));
cv::Mat finalFrame(frameHeight,frameWidth,CV_8UC3,Scalar::all(0));
cv::Mat detectFrame(frameHeight,frameWidth,CV_8UC3,Scalar::all(0));
cv::Mat trackingFrame(frameHeight,frameWidth,CV_8UC3,Scalar::all(0));
cv::Mat smallFrame(netSize,netSize,CV_8UC3,Scalar::all(0));
cv::Mat smallFrame32f(netSize,netSize,CV_32FC3,Scalar::all(0));

bool cameraReady = false;
int exitFlag = 0;

int fps = 0;
int fpsCam = 0;
std::time_t timeBegin = std::time(0);
int firstSample = 0;

Mat detections;
Mat detections_shadow;
int objectClass;
int startX;
int startY;
int endX;
int endY;
char label_text[128];
int label_left;
int label_top; 
int label_right;
int label_bottom;
Size label_size;

int base_index;
float confidence;
//unsigned int lenResultData = 1414;
//int numResults = lenResultData / sizeof(half);
//float* resultData32 = (float*)malloc(numResults * sizeof(*resultData32));

//detections.append([idx, confidence, x1, y1, x2, y2, label_text, label_left, label_top, label_right, label_bottom ])
struct detectionContainer {
  int objectClass;
  float confidence;
  int startX;
  int startY;
  int endX;
  int endY;
  char label_text[128];
  int label_left;
  int label_top; 
  int label_right;
  int label_bottom;
  Size label_size;
} ;

detectionContainer detection[20];
int validDetections = 0;
detectionContainer detectionShadow[20];
int validDetectionsShadow = 0;

int rotation = 100;
int xPos = 9;
int yPos = 8;
float grabPos = 1;
int currentTimestamp;
int activatedTimestamp;
int sendFlag = 0;

// function to create vector of class names
std::vector<String> createClassNames() {
    std::vector<String> classNames;
    classNames.push_back("background");
    classNames.push_back("aeroplane");
    classNames.push_back("bicycle");
    classNames.push_back("bird");
    classNames.push_back("boat");
    classNames.push_back("bottle");
    classNames.push_back("bus");
    classNames.push_back("car");
    classNames.push_back("cat");
    classNames.push_back("chair");
    classNames.push_back("cow");
    classNames.push_back("diningtable");
    classNames.push_back("dog");
    classNames.push_back("horse");
    classNames.push_back("motorbike");
    classNames.push_back("person");
    classNames.push_back("pottedplant");
    classNames.push_back("sheep");
    classNames.push_back("sofa");
    classNames.push_back("train");
    classNames.push_back("tvmonitor");
    return classNames;
} 

// function to create vector of colors
std::vector<cv::Scalar> createClassColors() {
    std::vector<cv::Scalar> classColors;
    classColors.push_back(cv::Scalar(  91,    6,   61));
    classColors.push_back(cv::Scalar( 175,  222,   26));
    classColors.push_back(cv::Scalar(  22,  189,   93));
    classColors.push_back(cv::Scalar( 234,   62,  135));
    classColors.push_back(cv::Scalar( 117,   87,  207));
    classColors.push_back(cv::Scalar( 213,  213,  132));
    classColors.push_back(cv::Scalar(  98,   54,  208));
    classColors.push_back(cv::Scalar( 249,  113,   45));
    classColors.push_back(cv::Scalar( 195,   18,  248));
    classColors.push_back(cv::Scalar( 199,   41,   95));
    classColors.push_back(cv::Scalar( 214,  118,  217));
    classColors.push_back(cv::Scalar( 140,   84,  153));
    classColors.push_back(cv::Scalar( 216,  250,  221));
    classColors.push_back(cv::Scalar( 219,  248,  169));
    classColors.push_back(cv::Scalar( 219,  205,   80));
    classColors.push_back(cv::Scalar(  21,   98,  240));
    classColors.push_back(cv::Scalar( 194,   74,  247));
    classColors.push_back(cv::Scalar( 104,  102,  209));
    classColors.push_back(cv::Scalar( 122,  118,   80));
    classColors.push_back(cv::Scalar( 198,  168,  155));
    classColors.push_back(cv::Scalar(  40,  160,  250));
    return classColors;
}

std::vector<String> classNames = createClassNames();
std::vector<cv::Scalar> classColors = createClassColors();

// Load a graph file
// caller must free the buffer returned.
void *LoadGraphFile(const char *path, unsigned int *length)
{
	FILE *fp;
	char *buf;

	fp = fopen(path, "rb");
	if(fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);
	*length = ftell(fp);
	rewind(fp);
	if(!(buf = (char*) malloc(*length)))
	{
		fclose(fp);
		return 0;
	}
	if(fread(buf, 1, *length, fp) != *length)
	{
		fclose(fp);
		free(buf);
		return 0;
	}
	fclose(fp);
	return buf;
}

void diep(const char *s)
{
  perror(s);
  exit(1);
}

void *task1(void *argument){
      char* msg;
      msg = (char*)argument;
      while(1){
        if (exitFlag) break;
        printf(msg);
        sleep(1);
      }
}

void timestamp() {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    printf("[%02d:%02d:%02d] ", tm.tm_hour, tm.tm_min, tm.tm_sec);
}

void *imageProcessing(void *data){
    timestamp();
    printf("Image processing thread started\n");
    
    int frameCounter = 0;
    int tick = 0;
    std::time_t timeNow;
    int detection_index;

    Net net;
    Mat inputBlob;
    //half *imgfp16 = (half*) malloc(sizeof(*imgfp16) * netSize * netSize * 3);
    int classId;
    double classProb;

    //void *deviceHandle;
    //char devName[NAME_SIZE];
    //void* graphHandle;
    ncStatus_t retCode;
    unsigned int optionSize;
    
    struct ncDeviceHandle_t* deviceHandle;
    struct ncGraphHandle_t* graphHandle;
    struct ncFifoHandle_t* inputFIFO = NULL;
    struct ncFifoHandle_t* outputFIFO = NULL;
    struct ncTensorDescriptor_t inputDescriptor;
    struct ncTensorDescriptor_t outputDescriptor;
    
    void* result;
    float *fresult;
    unsigned int fifoOutputSize = 0;
    unsigned int optionDataLen = sizeof(fifoOutputSize);
    
    float* imageData;
    unsigned int imageSize;
    //unsigned int lenBufFp16;
    //void* resultData16;
    //void* userParam;
    
    if (useMovidius==1) {
        // ncStatus_t status;
        // int loggingLevel = NC_LOG_DEBUG;
        // status = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loggingLevel, sizeof(int));
        // Initialize the device handle
        retCode = ncDeviceCreate(0, &deviceHandle);
        if (retCode != NC_OK)
        {   
            timestamp();
            // failed to get device name, maybe none plugged in.
            // Failed to initialize the device... maybe it isn't plugged in to the host
            printf("ncDeviceCreate Failed [%d]: Could not initialize the device.\n", retCode);
            timestamp();
            printf("Fall back to CPU use\n");
            useMovidius = 0;
            // exit(-1);
        }
    }
    if (useMovidius==1) {   

        retCode = ncDeviceOpen(deviceHandle);
        if (retCode != NC_OK)
        {
            timestamp();
            printf("ncDeviceOpen Failed [%d]: Could not open the device.\n", retCode);
            exit(-1);
        }
        else
        {
            // deviceHandle is ready to use now.  
            // Pass it to other NC API calls as needed and close it when finished.
            timestamp();
            printf("Successfully opened NCS device!\n");
        }
        // Now read in a graph file
        unsigned int graphBufferLength = 0;
        void* graphBuffer = LoadGraphFile(graphFileName, &graphBufferLength);
        // Initialize and allocate the graph to the device
        // Initialize the graph and give it a name.
        retCode = ncGraphCreate("My Graph", &graphHandle);
        if (retCode != NC_OK)
        {   // error allocating graph
            timestamp();
            printf("Error creating graph[%d]\n", retCode);
            exit(-1);
        }
        else
        {    
            // Graph is initialized. User can now call ncGraphAllocate and then
            // other API functions with the graph handle.
            timestamp();
            printf("Graph created OK!\n");
        }
        retCode = ncGraphAllocate(deviceHandle, graphHandle, graphBuffer, graphBufferLength);
        free(graphBuffer);
        
        if (retCode != NC_OK)
        {   
            // Could not allocate graph 
            timestamp();
            printf("Could not allocate graph for file: %s\n", graphFileName);
            timestamp();
            printf("Error from ncGraphAllocate is: %d\n", retCode);
            exit(-1);
        }
        else
        {
            // Graph is allocated and can be used for inference now.
            timestamp();
            printf("Graph allocated OK!\n");
        }
        
        // Create an input FIFO
        retCode = ncFifoCreate("MY Input FIFO", NC_FIFO_HOST_WO, &inputFIFO);
        if (retCode != NC_OK)
        {
            printf("Error - Input Fifo Initialization failed!");
            exit(-1);
        }
        optionSize = sizeof(ncTensorDescriptor_t);
        ncGraphGetOption(graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputDescriptor, &optionSize);
        retCode = ncFifoAllocate(inputFIFO, deviceHandle, &inputDescriptor, 2);
        if (retCode != NC_OK)
        {
            timestamp();
            printf("Error - Input Fifo allocation failed!");
            exit(-1);
        }
        // Create an output FIFO 
        retCode = ncFifoCreate("MY Output FIFO", NC_FIFO_HOST_RO, &outputFIFO);
        if (retCode != NC_OK)
        {
            timestamp();
            printf("Error - Output Fifo Initialization failed!");
            exit(-1);
        }
        optionSize = sizeof(ncTensorDescriptor_t);
        ncGraphGetOption(graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputDescriptor, &optionSize);
        retCode = ncFifoAllocate(outputFIFO, deviceHandle, &outputDescriptor, 2);
        if (retCode != NC_OK)
        {
            timestamp();
            printf("Error - Output Fifo allocation failed!");
            exit(-1);
        }
        ncFifoGetOption(outputFIFO, NC_RO_FIFO_ELEMENT_DATA_SIZE, &fifoOutputSize, &optionDataLen);
        result = malloc(fifoOutputSize);
        
    }
    if (useMovidius==0){
        
        try {
            net = dnn::readNetFromCaffe(modelTxt, modelBin); 
        }
        catch (cv::Exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            if (net.empty()) {
                std::cerr << "Can't load network by using the following files: " << std::endl;
                std::cerr << "prototxt:   " << modelTxt << std::endl;
                std::cerr << "caffemodel: " << modelBin << std::endl;
                std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
                std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
                exit(-1);
            }
        }
    }
    
    int key = 0;
    int idxConf[4] = { 0, 0, 0, 2 };
    int idxClass[4] = { 0, 0, 0, 1 };
    int idxStartX[4] = { 0, 0, 0, 3 };
    int idxStartY[4] = { 0, 0, 0, 4 };
    int idxEndX[4] = { 0, 0, 0, 5 };
    int idxEndY[4] = { 0, 0, 0, 6 };
        
    while(key != 27) {
        if (exitFlag) break;
        if ((faceDetectionEna) && (cameraReady)) {
            pthread_mutex_lock(&lock);
            frame.copyTo(detectFrame);
            pthread_mutex_unlock(&lock);
            
            validDetections = 0;
            
            if (useMovidius==1){
                cv::resize(detectFrame,smallFrame, Size(netSize,netSize));
                smallFrame.convertTo(smallFrame32f, CV_32FC3);
                smallFrame32f = smallFrame32f - Scalar(127,127,127);
                smallFrame32f = smallFrame32f * 0.007843;
                //cv::imshow( "smallframe", smallFrame32f );
                                
                imageData = (float*)smallFrame32f.data;
                imageSize = sizeof(float) * netSize * netSize * 3; //3 = numChannels;
                // Write the image to the input FIFO
                retCode = ncFifoWriteElem(inputFIFO, imageData, &imageSize, 0);
                if (retCode != NC_OK)
                {   
                    // Could not write FIFO element
                    timestamp();
                    printf("Error writing FIFO element [%d]\n", retCode);
                    exit(-1);
                }
 
                // Queue the inference
                retCode = ncGraphQueueInference(graphHandle, &inputFIFO, 1, &outputFIFO, 1);
                if (retCode != NC_OK)
                {
                    timestamp();
                    printf("Error - Failed to queue Inference!");
                    exit(-1);
                }
                                
                retCode = ncFifoGetOption(outputFIFO, NC_RO_FIFO_ELEMENT_DATA_SIZE, &fifoOutputSize, &optionDataLen);
                if (retCode || optionDataLen != sizeof(unsigned int)){
                    timestamp();
                    printf("ncFifoGetOption failed, rc=%d\n", retCode);
                    exit(-1);
                }
                // Get the results from the output FIFO
                retCode = ncFifoReadElem(outputFIFO, result, &fifoOutputSize, NULL);
                if (retCode != NC_OK)
                {
                    timestamp();
                    printf("Error - Read Inference result failed!");
                    exit(-1);
                }

                fresult = (float*) result;
                                
                for (int i=0; i < fresult[0]; i++) {
                    base_index = 7+ i * 7;
                    if (std::isfinite(fresult[base_index]) && std::isfinite(fresult[base_index+1]) && (fresult[base_index+1] > 0) && std::isfinite(fresult[base_index+2]) && std::isfinite(fresult[base_index+3]) && std::isfinite(fresult[base_index+4]) && std::isfinite(fresult[base_index+5]) && std::isfinite(fresult[base_index+6])){
                        
                        // printf("%f, %f, %f, %f, %f, %f, %f\n", fresult[base_index], fresult[base_index+1], fresult[base_index+2], fresult[base_index+3], fresult[base_index+4], fresult[base_index+5], fresult[base_index+6]);
                        detection[validDetections].objectClass = fresult[base_index+1];
                        detection[validDetections].confidence = fresult[base_index+2];
                        detection[validDetections].startX = fresult[base_index+3] * frameWidth;
                        detection[validDetections].startY = fresult[base_index+4] * frameHeight;
                        detection[validDetections].endX = fresult[base_index+5] * frameWidth - detection[validDetections].startX;
                        detection[validDetections].endY = fresult[base_index+6] * frameHeight - detection[validDetections].startY;

                        sprintf(label_text, "%s: %.2f%%", (classNames[detection[validDetections].objectClass]).c_str(), (detection[validDetections].confidence * 100) );
                        std::copy(label_text, label_text+128, detection[validDetections].label_text);
                        //printf("%s\n", label_text);
                        detection[validDetections].label_size = getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1,0);
                        detection[validDetections].label_left = detection[validDetections].startX;
                        label_top = detection[validDetections].startY - detection[validDetections].label_size.height;
                        if (label_top < 1){
                            label_top = 1;
                        }
                        detection[validDetections].label_top = label_top;
                        detection[validDetections].label_right = detection[validDetections].label_size.width;
                        detection[validDetections].label_bottom = detection[validDetections].label_size.height;

                        validDetections++;
                        
                    }
                }
                
            }
            // use movidius = 0 -> CPU net
            else {          
                inputBlob = blobFromImage(detectFrame, 0.007843, Size(netSize, netSize), Scalar(127.5,127.5,127.5));   //Convert Mat to batch of images
                net.setInput(inputBlob);        //set the network input
                detections = net.forward();             //compute output
                
                // look what the detector found
                for (int i=0; i < detections.size[2]; i++) {

                    // confidence
                    idxConf[2] = i;
                    detection[validDetections].confidence = detections.at<float>(idxConf);                    

                    // detected class
                    idxClass[2] = i;
                    detection[validDetections].objectClass = detections.at<float>(idxClass);

                    // bounding box
                    idxStartX[2] = i;
                    idxStartY[2] = i;
                    idxEndX[2] = i;
                    idxEndY[2] = i;
                    detection[validDetections].startX = detections.at<float>(idxStartX) * frameWidth;
                    detection[validDetections].startY = detections.at<float>(idxStartY) * frameHeight;
                    detection[validDetections].endX = detections.at<float>(idxEndX) * frameWidth - detection[validDetections].startX;
                    detection[validDetections].endY = detections.at<float>(idxEndY) * frameHeight - detection[validDetections].startY;
                    
                    sprintf(label_text, "%s: %.2f%%", (classNames[detection[validDetections].objectClass]).c_str(), (detection[validDetections].confidence * 100) );
                    std::copy(label_text, label_text+128, detection[validDetections].label_text);
                    //printf("%s\n", label_text);
                    
                    detection[validDetections].label_size = getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1,0);
                    detection[validDetections].label_left = detection[validDetections].startX;
                    label_top = detection[validDetections].startY - detection[validDetections].label_size.height;
                    if (label_top < 1){
                        label_top = 1;
                    }
                    detection[validDetections].label_top = label_top;
                    detection[validDetections].label_right = detection[validDetections].label_size.width;
                    detection[validDetections].label_bottom = detection[validDetections].label_size.height;

                    validDetections++;
                }
                
            }
            
            firstSample = 1;
            pthread_mutex_lock(&lock);
            validDetectionsShadow = validDetections;
            std::copy(detection, detection+20, detectionShadow);
            pthread_mutex_unlock(&lock);

            frameCounter++;
            timeNow = std::time(0) - timeBegin;
            if (timeNow - tick >= 1) {
                tick++;
                fps = frameCounter;
                frameCounter = 0;
            }

            // usleep(30000);
        }
        else {
            sleep(1);
        }
    }
    timestamp();
    // Clean up
    retCode = ncFifoDestroy(&inputFIFO);
    retCode = ncFifoDestroy(&outputFIFO);
    retCode = ncGraphDestroy(&graphHandle);
    ncDeviceClose(deviceHandle);
    ncDeviceDestroy(&deviceHandle);
    printf("Image processing thread stopped\n");
}

void *objectTracking(void *data){
    
    timestamp();
    printf("Object tracking thread started\n");
    
    if (trackerType == "BOOSTING")
        tracker = TrackerBoosting::create();
    if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    if (trackerType == "TLD")
        tracker = TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        tracker = TrackerGOTURN::create();
    if (trackerType == "MOSSE")
        tracker = TrackerMOSSE::create();
    
    bool success;

    
    while(1) {
        if (exitFlag) break;
        if ((objectTrackingEna) && (cameraReady)) {
            pthread_mutex_lock(&lock);
            frame.copyTo(trackingFrame);
            pthread_mutex_unlock(&lock);
            
            if (ROIset != 0){
                // Update the tracking result
                success = tracker->update(trackingFrame, outBox);
                
                pthread_mutex_lock(&lock);
                successShadow = success;
                outBoxShadow = outBox;
                pthread_mutex_unlock(&lock);
            }
            else {
                sleep(1);
            }
        }
        else{
            sleep(1);
        }
    }
    timestamp();
    printf("Object tracking thread stopped\n");
    
}

void *visualizer(void *data){
    timestamp();
    printf("Visualizer thread started\n");
    
    int key = 0;
    
    int frameCounter = 0;
    int tick = 0;
    std::time_t timeNow;
    
    cv::namedWindow( "result", cv::WINDOW_NORMAL );
    cv::setWindowProperty("result", cv::WND_PROP_FULLSCREEN,cv::WINDOW_FULLSCREEN);
    
    
    while(key != 27) {
        if (exitFlag) break;
        
        pthread_mutex_lock(&lock);
        frame.copyTo(finalFrame);
        pthread_mutex_unlock(&lock);
        
        if (withScreen){
            if (firstSample == 1){                
                pthread_mutex_lock(&lock);
                for (int i=0; i < validDetectionsShadow; i++) {
                    if (detectionShadow[i].confidence > confidenceThreshold) {
                        rectangle(finalFrame, cv::Rect(detectionShadow[i].startX, detectionShadow[i].startY, detectionShadow[i].endX, detectionShadow[i].endY), classColors[detectionShadow[i].objectClass],1,8,0);
                        //if (detectionShadow[i].objectClass == 15){
                        //    line(finalFrame, Point((detectionShadow[i].endX/2 + detectionShadow[i].startX),0), Point((detectionShadow[i].endX/2 + detectionShadow[i].startX), frameHeight), cv::Scalar(0, 0, 255),1);
                        //    line(finalFrame, Point(0,(detectionShadow[i].endY/2 + detectionShadow[i].startY)), Point(frameWidth, (detectionShadow[i].endY/2 + detectionShadow[i].startY)), cv::Scalar(0, 0, 255),1);
                        //}
                        
                        rectangle(finalFrame, cv::Rect(detectionShadow[i].label_left -1, detectionShadow[i].label_top -1, detectionShadow[i].label_right +1, detectionShadow[i].label_bottom +1), cv::Scalar(125, 175,75),-1);
                        cv::putText(finalFrame, detectionShadow[i].label_text, cv::Point(detectionShadow[i].label_left, detectionShadow[i].label_top + detectionShadow[i].label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255),1);
                    }
                }
                pthread_mutex_unlock(&lock);
                cv::putText(finalFrame, cv::format("Neural network FPS: %d", fps ), cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0));
            }
            
            if (ROIset){
                pthread_mutex_lock(&lock);
                if (successShadow){
                    rectangle(finalFrame, outBoxShadow, cv::Scalar(0, 255,255),2,8,0);
                }
                pthread_mutex_unlock(&lock);
            }
            
            frameCounter++;
            timeNow = std::time(0) - timeBegin;
            if (timeNow - tick >= 1) {
                tick++;
                fpsCam = frameCounter;
                frameCounter = 0;
            }
            cv::putText(finalFrame, cv::format("Display FPS: %d", fpsCam ), cv::Point(5, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0));
            cv::imshow( "result", finalFrame );
        
            if (key == 115){
                inputROI = cv::selectROI("result", finalFrame, true, false);
                tracker->init(finalFrame, inputROI);
                ROIset = 1;
            }
        
        }
        key=cv::waitKey( 10 );

        
    }
    exitFlag = 1;
    timestamp();
    printf("Visualizer thread stopped\n");
}

void *cameraCapture(void *data){
    timestamp();
    printf("Capturing thread started\n");

    //VideoCapture Camera(cameraIndex);
    raspicam::RaspiCam_Cv Camera;
    //Camera.setVerticalFlip(1);
    //Camera.setHorizontalFlip(1);
    Camera.setRotation(270);
    
    //if(!Camera.open(cameraIndex)) diep("Capture from CAM didn't work");
    
    //Camera.set( cv::CAP_PROP_FRAME_WIDTH, frameWidth );
    //Camera.set( cv::CAP_PROP_FRAME_HEIGHT, frameHeight );
    //Camera.set( cv::CAP_PROP_FPS, 20.0);

    // Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 );
    
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, frameWidth );
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, frameHeight );
  
    if(!Camera.open()) diep("Capture from CAM didn't work");
  
    timestamp();
    printf("Successfully connected to camera\n");
    
    int key = 0;
    
    while(key != 27) {
        if (exitFlag) break;
        
        Camera.grab();
        pthread_mutex_lock(&lock);
        Camera.retrieve ( frame );
        pthread_mutex_unlock(&lock);
        cameraReady = true;
        usleep(10000);
        // key=cv::waitKey( 1 );
    }
    exitFlag = 1;
    Camera.release();
    timestamp();
    printf("Capturing thread stopped\n");

}

void *uartHandler(void *data){
    timestamp();
    printf("UART handler thread started\n");
    int fd = *((int *)data);
    char textBuf[30];
    
    int xPosUART = 0;
    int yPosUART = 0;
    int grabUART = 100;
    while (1){
        if (exitFlag) break;
        xPosUART = (int)(xPos*10);
        yPosUART = (int)((yPos*10)+50);
        grabUART = (int)(100*grabPos);
        if (sendFlag == 1){
            pthread_mutex_lock(&lock);
            //sprintf(textBuf, "S%03d;%03d;%03d;%03d\r\n",  rotation, xPosUART, yPosUART, grabUART);
            sprintf(textBuf, "P%03d;%03d;%03d\r\n",  rotation, xPosUART, yPosUART);
            //sprintf(textBuf, "P%03d;%03d;%03d\r\n",  100, xPosUART, yPosUART);
            pthread_mutex_unlock(&lock);
            serialPuts(fd, textBuf);
            sendFlag = 0;
        }
        //usleep(50000);
        sleep(1);

    }
    timestamp();
    printf("UART handler thread stopped\n");
}

void *armMovement(void *data){
    timestamp();
    printf("Arm movement thread started\n");
    
    int FACEBUFFERSIZE = 20;
    
    int faceXPositionBuffer[FACEBUFFERSIZE];
    int faceYPositionBuffer[FACEBUFFERSIZE];
    int faceWidthBuffer[FACEBUFFERSIZE];
    int faceTimeBuffer[FACEBUFFERSIZE];
    
    int n = 0;
    int numberOfSamples = 0;
    int sumOfX = 0;
    int avgOfX = 0;
    int state = 0;
    int firstRun = 0;
    int biggestFace = 0;
    
    memset(faceXPositionBuffer, 0, FACEBUFFERSIZE*sizeof(int));
    memset(faceWidthBuffer, 0, FACEBUFFERSIZE*sizeof(int));
    memset(faceTimeBuffer, 0, FACEBUFFERSIZE*sizeof(int));
    
    
    while(1) {
        if (exitFlag) break;
        if (faceDetectionEna) {
            if (state == 0){ // detecting faces
                biggestFace = 0;
                for (int i=0; i < validDetectionsShadow; i++) {
                    if ((detectionShadow[i].confidence > confidenceThreshold) && (detectionShadow[i].objectClass == 15)) {
                        //printf("%d -- %d\n", detectionShadow[i].startX, detectionShadow[i].endX);
                        if (detectionShadow[i].endX > biggestFace){
                            faceWidthBuffer[n] = detectionShadow[i].endX;
                            faceXPositionBuffer[n] = detectionShadow[i].endX/2 + detectionShadow[i].startX;
                            //printf("%d\n",faceXPositionBuffer[n]);
                            faceTimeBuffer[n] = (int)time(NULL);
                            biggestFace = detectionShadow[i].endX;
                        }
                    }
                }
                currentTimestamp = (int)time(NULL);
                numberOfSamples = 0;
                sumOfX = 0;
                
                //printf("size: %d\n",biggestFace);
                
                for(int i=0; i<FACEBUFFERSIZE; i++){

                    //printf("diff: %d\n",(currentTimestamp-faceTimeBuffer[i]));
                    if ((currentTimestamp-faceTimeBuffer[i] < 5) && (biggestFace != 0)){
                        numberOfSamples++;
                        sumOfX += faceXPositionBuffer[i];
                    }
                }
                if ((numberOfSamples > 6) && (armMovementEna)){
                    avgOfX = sumOfX/numberOfSamples;
                    state = 1;
                    firstRun = 1;
                    activatedTimestamp = (int)time(NULL);
                    timestamp();
                    printf("State changed to 1\n");
                }
                n++;
                if (n>FACEBUFFERSIZE){
                    n = 0;
                }
            }
            else if (state == 1){ // turn to the face and wait
                xPos = 21;
                yPos = 15;
                grabPos = 0.5;
                if (firstRun == 1){
                    sendFlag = 1;
                    firstRun = 0;
                }
                
                rotation = 100 + (avgOfX-frameWidth/2)*0.6; 
                
                if (rotation < 0) {
                    rotation = 0;
                }
                else if (rotation > 200){
                    rotation = 200;
                }
                
                currentTimestamp = (int)time(NULL);
                if (currentTimestamp-activatedTimestamp > 8){
                    state = 2;
                    firstRun = 1;
                    activatedTimestamp = (int)time(NULL);
                    timestamp();
                    printf("State changed to 2\n");

                }
            }
            else if (state == 2){ // go back to init state and wait
                rotation = 100;
                xPos = 9;
                yPos = 8;
                grabPos = 1;
                
                if (firstRun == 1){
                    sendFlag = 1;
                    firstRun = 0;
                }
                
                currentTimestamp = (int)time(NULL);
                if (currentTimestamp-activatedTimestamp > 8){
                    n = 0;
                    state = 0;
                    timestamp();
                    printf("State changed to 0\n");

                }
            }
            usleep(300000);
        }
    }
    timestamp();
    printf("Arm movement thread stopped\n");
}

int main(void) {
    int uartfd;
    
    pthread_t thread1, thread2, thread3, thread4, thread5, thread6;
    int i1,i2,i3, i4, i5, i6;
    
    i1 = pthread_create( &thread1, NULL, cameraCapture, (void*) 0);
    i2 = pthread_create( &thread2, NULL, imageProcessing, (void*) 0);    
    i3 = pthread_create( &thread3, NULL, visualizer, (void*) 0);    
    i4 = pthread_create( &thread4, NULL, armMovement, (void*) 0);    
    i6 = pthread_create( &thread6, NULL, objectTracking, (void*) 0);    

    if (withUART){
        timestamp();
        printf("Setting up UART\n");
        if ((uartfd = serialOpen ("/dev/ttyACM0", 9600)) < 0) diep("UART");
        i5 = pthread_create( &thread5, NULL, uartHandler, (void*) &uartfd);
    }
    
    while(1) {
        
        if (exitFlag) break;        
        sleep(1);
  
    } /* end of while */
    
    timestamp();
    printf("Application has gracefully stopped\n");
    return 0;
}



