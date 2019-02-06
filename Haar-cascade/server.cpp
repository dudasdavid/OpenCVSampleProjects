#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
// #include <fcntl.h>
#include <arpa/inet.h>

#include <time.h>

#include <pthread.h>
#include <wiringPiI2C.h>
#include <wiringPi.h> 
#include <wiringSerial.h> 


#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "RaspiCamCV.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

#define BUFLEN 9//512
#define TXBUFLEN 76800
#define NPACK 10
#define PORT 7777

#define dID 0x4A

/*
#define MAX_PEOPLE 	6
#define P_ARNOLD    0
#define P_GEORGE	1
#define P_KEANU 	2
#define P_LISA  	3
#define P_DAVID  	4
#define P_JENO  	5
*/
#define MAX_PEOPLE 	2
#define P_DAVID     0
#define P_DAVIDG	1

#define CONFIDENCE_MAX 2800
#define SERVO_THRESHOLD 20

unsigned char i2cBuffer[6];
int batteryRaw = 0;
double batteryFloat = 0.0;
char temperature = 35;
int S1 = 500, S2 = 800;
int S1last = 500, S2last = 500;
int S1limited = 500, S2limited = 500;

int LeftDrive = 100, RightDrive = 100;

int LEDState = 0;
int LEDState2 = 0;
int LEDStateLast = 0;
int LEDStateLast2 = 0;
int faceDetectionEna = 0;
int firstConnection = 1;
int messageCounter = 0;
int messageCounterAge = 0;
int TCPTimeout = 0;

int trackingSwitch = 0;
int saveFacesSwitch = 0;

int scaleSelector = 0;

int picIndex = 1;

pthread_mutex_t lock;

vector<Rect> faces;
CascadeClassifier cascade;

String csvName = "/home/pi/Documents/OpenCVResources/Faces/faces.csv";
String cascadeName = "/home/pi/Documents/OpenCVResources/Cascade/lbpcascade_frontalface.xml";

String facePositions;
String faceNames;
int facesDetected = 0;
int faceAge = 0;

Mat frame, roi;
Mat sendFrame( 240, 320, CV_8UC1 );

struct sockaddr_in si_me, si_other;

int exitFlag = 0;
int tcpErrorFlag = 0;

vector<Mat> images;
vector<int> labels;
int im_width;
int im_height;
// Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
// Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
string people[MAX_PEOPLE];

char savedImageName[100];
int imageCounter = 0;

void diep(const char *s)
{
  perror(s);
  exit(1);
}

void cleanString(volatile char *s);
void servoMessage(uint16_t s1, uint16_t s2);
char measureTemp(void);
void doprocessing (int sock);
char ledMessage(int val);
string int_array_to_string(int int_array[], int size_of_array);
void faceDetect( Mat gray, CascadeClassifier& cascade );
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
int GetCPULoad();

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

void *detectFaceOnFrame(void *data){
    timestamp();
    printf("Face detection thread started\n");
    
    if(!cascade.load( cascadeName )) diep("Could not load classifier cascade");
    timestamp();
    printf("LBP Cascade loaded\n");
    
    
    try {
        read_csv(csvName, images, labels, ';');
    } 
    catch (cv::Exception& e) {
        cerr << "Error opening file \"" << csvName << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    im_width = images[0].cols;
    im_height = images[0].rows;
    model->train(images, labels);
    /*
    people[P_ARNOLD] 	= "Arnold";
    people[P_GEORGE]  	= "George";
    people[P_KEANU]  	= "Keanu";
    people[P_LISA]  	= "Lisa";
    people[P_DAVID]  	= "David";
    people[P_JENO]  	= "Jeno";
    */
    people[P_DAVID] 	= "David";
    people[P_DAVIDG]  	= "David";
    
    timestamp();
    printf("Face database loaded\n");
    
    
    sleep(1);
    Mat detectFrame;
    while(1) {
        if (exitFlag) break;
        if (faceDetectionEna) {
            pthread_mutex_lock(&lock);
            frame.copyTo(detectFrame);
            pthread_mutex_unlock(&lock);
            faceDetect(detectFrame, cascade);
        }
        else {
            sleep(1);
        }
    }
    timestamp();
    printf("Face detection thread stopped\n");
}

void *cameraCapture(void *data){
    timestamp();
    printf("Capturing thread started\n");
    RaspiCamCvCapture* capture = 0;
    
    capture = raspiCamCvCreateCameraCapture( 0, 640, 480, 30, 1 ); /* source (0), width (320), height (240), fps (15), grayscale (1)*/
    if(!capture) diep("Capture from CAM didn't work");
    timestamp();
    printf("Successfully connected to camera\n");
    
    while(1) {
        if (exitFlag) break;
        pthread_mutex_lock(&lock);
        frame = raspiCamCvQueryFrame( capture );
        pthread_mutex_unlock(&lock);
        cvWaitKey( 1 );
    }
    raspiCamCvReleaseCapture( &capture );
    timestamp();
    printf("Capturing thread stopped\n");
}

void *uartHandler(void *data){
    timestamp();
    printf("UART handler thread started\n");
    int fd = *((int *)data);
    while (1){
        if (exitFlag) break;
        
        serialPutchar(fd, LeftDrive & 0xFF);
        serialPutchar(fd, RightDrive & 0xFF);
        serialPutchar(fd, 0xFF);
        
        usleep(35000);
        
        if (LEDState2 != LEDStateLast2) {
            LEDStateLast2 = LEDState2;
            serialPutchar(fd, LEDState2 & 0xFF);
            serialPutchar(fd, LEDState2 & 0xFF);
            serialPutchar(fd, 0xFE);
            // ledValue = ledMessage(LEDState2);
            // wiringPiI2CWriteReg8 (fd, 0xAD, ledValue);
            // usleep(10000);
        }
        
        usleep(35000);
    }
    timestamp();
    printf("UART handler thread stopped\n");
}

void *i2cHandler(void *data){
    timestamp();
    printf("I2C handler thread started\n");
    int a = 0;
    int fd = *((int *)data);
    unsigned char ledValue = 0;
    while (1){
        if (exitFlag) break;
        if (a % 100 == 0) {
            batteryRaw = wiringPiI2CReadReg8 (fd, 0xAC);
            batteryFloat = ((batteryRaw & 0xF0) >> 4) + ((batteryRaw & 0x0F) / 10.0);
            temperature = measureTemp();
            wiringPiI2CWriteReg8 (fd, 0xAB, temperature);
        }
        
        a++;
        
        if ((S1 != S1last) || (S2 != S2last)) {
            //printf("%d,%d,%d,%d\n", S1, S1last, S2, S2last);
            if (S1 > S1last + SERVO_THRESHOLD) {
                S1limited += SERVO_THRESHOLD;
            }
            else if (S1 < S1last - SERVO_THRESHOLD) {
                S1limited -= SERVO_THRESHOLD;
            }
            else S1limited = S1;
            
            S1last = S1limited;
            S2last = S2;
            servoMessage(S2,S1limited);
            write(fd, i2cBuffer, 6);
            // usleep(10000);
        }
        if (LEDState != LEDStateLast) {
            LEDStateLast = LEDState;
            ledValue = ledMessage(LEDState);
            wiringPiI2CWriteReg8 (fd, 0xAD, ledValue);
            // usleep(10000);
        }
        //printf("LED: %d\n", LEDState);
        
        usleep(50000);
        // sleep(1);
    }
    timestamp();
    printf("I2C handler thread stopped\n");
}

void *sendTCP(void *data){
    timestamp();
    printf("TCP send thread started\n");
    int e;
    int sock = *((int *)data);
    int n = 0;
    char textBuffer[10];
    char faceBuffer[100];
    
    // unsigned char array[76800] = { 0 };
    // for (int i=0; i<76800; i++) {
        // array[i] = rand()%255;
    // }
    unsigned char *array;
    sleep(1);
    while(1) {
        if (exitFlag || tcpErrorFlag) break;
        usleep(50000);
        //sleep(1);
        // printf("size: %d * %d\n", sendFrame.rows, sendFrame.cols);
        
        e = send(sock,"image",5, MSG_NOSIGNAL);
        if (e < 0) break;
        
        pthread_mutex_lock(&lock);
        if (!scaleSelector) resize( frame, sendFrame, sendFrame.size(), 0, 0, INTER_LINEAR );
        else {
            roi = frame(Rect(frame.cols/2 - 160,frame.rows/2 - 120,320,240));
            roi.copyTo(sendFrame);
        }
        pthread_mutex_unlock(&lock);
        
        array = sendFrame.data;

        e = send(sock,array,76800,MSG_NOSIGNAL);
        if (e < 0) break;
        
        if (firstConnection) {
            firstConnection = 0;
            cleanString(textBuffer);
            e = send(sock,"dflt_",5,MSG_NOSIGNAL);
            if (e < 0) break;
            sprintf(textBuffer, "%d;%d", S1, S2);
            e = send(sock,textBuffer,10,MSG_NOSIGNAL);
            if (e < 0) break;
        }

        if (n % 100 == 0) {
            cleanString(textBuffer);
            e = send(sock,"temp_",5,MSG_NOSIGNAL);
            if (e < 0) break;
            sprintf(textBuffer, "%d", temperature);
            e = send(sock,textBuffer,10,MSG_NOSIGNAL);
            if (e < 0) break;
            
            cleanString(textBuffer);
            e = send(sock,"batt_",5,MSG_NOSIGNAL);
            if (e < 0) break;
            sprintf(textBuffer, "%.1f", batteryFloat);
            e = send(sock,textBuffer,10,MSG_NOSIGNAL);
            if (e < 0) break;
            
            if (messageCounter == messageCounterAge){
                e = send(sock,"alive",5,MSG_NOSIGNAL);
                if (e < 0) break;
                TCPTimeout++;
                if (TCPTimeout > 3) tcpErrorFlag = 1;
            }
            else {
                messageCounterAge = messageCounter;
                TCPTimeout = 0;
            }
        }

        if (facesDetected){
            cleanString(faceBuffer);
            e = send(sock,"faces",5,MSG_NOSIGNAL);
            if (e < 0) break;
            sprintf(faceBuffer, "%s", facePositions.c_str());
            e = send(sock,faceBuffer,100,MSG_NOSIGNAL);
            if (e < 0) break;
            
            cleanString(faceBuffer);
            e = send(sock,"names",5,MSG_NOSIGNAL);
            if (e < 0) break;
            sprintf(faceBuffer, "%s", faceNames.c_str());
            e = send(sock,faceBuffer,100,MSG_NOSIGNAL);
            if (e < 0) break;
            
            facesDetected = 0;
            //printf("%s\n",facePositions.c_str());
        }
        
        n++;
        
        
    }
    timestamp();
    printf("TCP send thread stopped\n");
    tcpErrorFlag = 1;
}

void *receiveTCP(void *data){
    timestamp();
    printf("TCP receive thread started\n");
    char buf[BUFLEN];
    int e;
    int sock = *((int *)data);

    while(1){
        if (exitFlag || tcpErrorFlag) break;
        cleanString(buf);
        e = read(sock, buf, BUFLEN);
        if (e <= 0) break;
        messageCounter++;
        
        timestamp();
        printf("Received %d bytes from %s:%d; Data: %s\n",
            e, inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port), buf);
            
        if ((buf[0] == 'S') and (0 == trackingSwitch)) {
            S1 = (buf[1]  - '0')*1000 + (buf[2]  - '0')*100 + (buf[3]  - '0')*10 + (buf[4]  - '0')*1;
            S2 = (buf[5]  - '0')*1000 + (buf[6]  - '0')*100 + (buf[7]  - '0')*10 + (buf[8]  - '0')*1;
        }
        else if (buf[0] == 'Z') scaleSelector = (buf[1]  - '0')*1;       
        else if (buf[0] == 'L') LEDState = (buf[1]  - '0')*10 + (buf[2]  - '0')*1;
        else if (buf[0] == 'R') LEDState2 = (buf[1]  - '0')*10 + (buf[2]  - '0')*1;
        else if (buf[0] == 'X') exitFlag = 1;
        else if (buf[0] == 'Q') tcpErrorFlag = 1;
        else if (buf[0] == 'A') TCPTimeout = 0;
        
        else if (buf[0] == 'D') {
            LeftDrive  = (buf[1]  - '0')*100 + (buf[2]  - '0')*10 + (buf[3]  - '0')*1;
            RightDrive = (buf[4]  - '0')*100 + (buf[5]  - '0')*10 + (buf[6]  - '0')*1;
            printf("Timestamp: %d\n",(int)time(NULL));
        }
        
        else if (buf[0] == 'F') {
            faceDetectionEna = (buf[1]  - '0')*1;
            if (!faceDetectionEna) {
                int facePositionBuffer[15];
                memset(facePositionBuffer, 0, 15*sizeof(int));
                facePositions = int_array_to_string(facePositionBuffer,15);
                facesDetected = 1;
            }
        }
    }
    
    timestamp();
    printf("TCP receive thread stopped\n");
    tcpErrorFlag = 1;
}

int main(void) {
    char buf[BUFLEN];
    int i2cfd, uartfd;
    int sockfd, i, newsockfd, pid;
    socklen_t slen=sizeof(si_other);
    
    pthread_t thread1, thread2, thread3, thread4, thread5, thread6;
    int i1,i2,i3,i4,i5,i6;
    int tr=1;
    
    timestamp();
    printf("Setting up UART\n");
    
    if ((uartfd = serialOpen ("/dev/ttyAMA0", 115200)) < 0) diep("UART");
    // if (wiringPiSetup () == -1) diep("wiringPi");
    
    timestamp();
    printf("Setting up I2C\n");
    i2cfd=wiringPiI2CSetup(dID);
    if(i2cfd < 0) diep("I2C");
    
    timestamp();
    printf("Creating socket\n");
    if ((sockfd=socket(AF_INET, SOCK_STREAM, 0))==-1) 
        diep("socket");

    memset((char *) &si_me, 0, sizeof(si_me));
    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(PORT);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);
    
    // kill "Address already in use" error message
    if (setsockopt(sockfd,SOL_SOCKET,SO_REUSEADDR,&tr,sizeof(int)) == -1)
        diep("setsockopt");

    
    if (bind(sockfd, (struct sockaddr *) &si_me, sizeof(si_me))==-1)
        diep("bind");
    
    listen(sockfd,5); 
  
    i1 = pthread_create( &thread1, NULL, i2cHandler, (void*) &i2cfd);
    i6 = pthread_create( &thread6, NULL, uartHandler, (void*) &uartfd);
    i4 = pthread_create( &thread4, NULL, cameraCapture, (void*) 0);
    i5 = pthread_create( &thread5, NULL, detectFaceOnFrame, (void*) 0);
    
    sleep(2);
    
    struct timeval tv;
    tv.tv_sec  = 30;  
    tv.tv_usec = 0;
    
    while(1) {
        
        if (exitFlag) break;
        
        tcpErrorFlag = 0;
        timestamp();
        printf("Waiting for connection...\n");
        newsockfd = accept(sockfd, (struct sockaddr *) &si_other, &slen);
        if (newsockfd < 0) diep("accept");
        
        
        setsockopt( newsockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        
        timestamp();
        printf("%s:%d connected to the server\n", inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port));
      
        //close(sockfd);
        i2 = pthread_create( &thread2, NULL, sendTCP, (void*) &newsockfd);
        i3 = pthread_create( &thread3, NULL, receiveTCP, (void*) &newsockfd);
        while (1) {
            if (tcpErrorFlag) break;
            sleep(1);
        }
        S1 = 500;
        S2 = 900;
        scaleSelector = 0;
        LEDState = 0;
        faceDetectionEna = 0;
        firstConnection = 1;
  
    } /* end of while */
    
    timestamp();
    printf("Reset everything to the default state\n");
    wiringPiI2CWriteReg8 (i2cfd, 0xAD, ledMessage(0));
    servoMessage(500,500);
    write(i2cfd, i2cBuffer, 6);
    
    timestamp();
    printf("Server has gracefully stopped\n");
    return 0;
}

void cleanString(volatile char *s) {
  int i = 0;
  while(s[i] != '\0'){                       // replace every character with \0
    s[i] = '\0';
    i++;
  }
}

void servoMessage(uint16_t s1, uint16_t s2) {
    if (S1 < 0) S1 = 0;
    if (S2 < 0) S2 = 0;
    if (S1 > 1000) S1 = 1000;
    if (S2 > 1000) S2 = 1000;
    
    int checksum;
	i2cBuffer[0] = 0xAA;
    i2cBuffer[1] = (s1 & 0xff00) >> 8;
    i2cBuffer[2] = (s1 & 0x00ff);
    i2cBuffer[3] = (s2 & 0xff00) >> 8;
    i2cBuffer[4] = (s2 & 0x00ff);
    checksum = i2cBuffer[1] + i2cBuffer[2] + i2cBuffer[3] + i2cBuffer[4];
    i2cBuffer[5] = (checksum & 0x00ff);
}

char measureTemp(void) {
    FILE *temperatureFile;
    int T;
    temperatureFile = fopen ("/sys/class/thermal/thermal_zone0/temp", "r");
    if (temperatureFile == NULL)
      ; //print some message
    fscanf (temperatureFile, "%d", &T);
    fclose (temperatureFile);
    return T/1000;
}

char ledMessage(int val) {
    return (char) ((((val & 0x0f) << 4) | (~val & 0x0f)) & 0xff);
}

string int_array_to_string(int int_array[], int size_of_array) {
  ostringstream oss("");
  for (int temp = 0; temp < size_of_array; temp++)
    oss << int_array[temp] << ";";
  return oss.str();
}

void faceDetect( Mat gray, CascadeClassifier& cascade )
{
    int i = 0;
    double t = 0;
    Point center;
    Point origo;
    Point p1;
    Point p2;
    int radius;
    char textBuffer[11];
    

    // equalizeHist( gray, gray );
 
    
    cascade.detectMultiScale( gray, faces, 1.3, 3, 0, Size(25, 25));
      
    if (faces.size() != 0) {
        faceAge = 0;
        int facePositionBuffer[15];
        memset(facePositionBuffer, 0, 15*sizeof(int));
        string box_text = "";
        
        if (saveFacesSwitch == 1) {
            sprintf(textBuffer, "test_%02d.jpg", picIndex);
            picIndex ++;
            imwrite(textBuffer, gray);
        }
        
        for(int i = 0; i < faces.size(); i++)
        //for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++)
        {
            if (i > 4) break;
            Rect face_i = faces[i];
            
            facePositionBuffer[0 + i*3] = faces[i].x;
            facePositionBuffer[1 + i*3] = faces[i].y;
            facePositionBuffer[2 + i*3] = faces[i].width;
            

            Mat face_resized;
            Mat face = gray(face_i);
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // int prediction = model->predict(face_resized);
            
            /*
            sprintf(savedImageName, "../../savedfaces/face%d.jpg", imageCounter++);
            imwrite(savedImageName, face_resized);
            */
            
            double confidence = 0.0;
            int label;
            int prediction;
            // int prediction = model->predict(face_resized, label,confidence);
            model->predict(face_resized, prediction, confidence);
            printf("!!%f!!\n", confidence);
            // = people[prediction]; //format("Prediction = %d", prediction);
            if (CONFIDENCE_MAX > confidence) box_text = box_text + people[prediction] + ";";
            else box_text = box_text + "Unknown" + ";";
            //printf("%s\n",box_text.c_str());
            //int pos_x = std::max(face_i.tl().x - 10, 0);
            //int pos_y = std::max(face_i.tl().y - 10, 0);
            //putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, color, 2.0);
            
            if ((1 == trackingSwitch) && (P_DAVID == prediction) && (CONFIDENCE_MAX > confidence)) {
                // printf("%d;%d\n", faces[i].x, faces[i].y);
                
                // S1 = 500 - int((320 - faces[i].x - (faces[i].width / 2.0))*1.0);
                S1 = S1 - int((320 - faces[i].x - (faces[i].width / 2.0))*0.4);
                // S2 = 500 - int((240 - faces[i].y - (faces[i].width / 2.0))*0.75);
                S2 = S2 - int((240 - faces[i].y - (faces[i].width / 2.0))*0.4);
                // printf("%d:%d\n", int((320 - faces[i].x - (faces[i].width / 2.0))*0.3), S1);
            }
        }
        if (faceDetectionEna) {
            facePositions = int_array_to_string(facePositionBuffer,15);
            faceNames = box_text;
            facesDetected = 1;
            //printf("%s\n",facePositions.c_str());
        }
    }
    else {
        faceAge++;
        if (faceAge == 5) {
            int facePositionBuffer[15];
            memset(facePositionBuffer, 0, 15*sizeof(int));
            facePositions = int_array_to_string(facePositionBuffer,15);
            facesDetected = 1;
        }
    }
        
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "(E) No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;

    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
        	// read the file and build the picture collection
            // printf("%s",path);
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int GetCPULoad() {
    FILE *cpuFile;
    float load;
    cpuFile = fopen("/proc/loadavg", "r");
    if (cpuFile == NULL) return -1;
    fscanf (cpuFile, "%f", &load);
    fclose (cpuFile);
    return (int)(load * 100);
   
}