all: run

run: server
	./server
server: server.cpp
	g++ server.cpp -o server \
	-lpthread \
    `pkg-config opencv --libs` -I/usr/local/include \
	-I$(HOME)/git/robidouille/raspicam_cv \
	-L$(HOME)/git/wiringPi/wiringPi -lwiringPi \
	-L$(HOME)/git/robidouille/raspicam_cv -lraspicamcv \
	-L$(HOME)/git/raspberrypi/userland/build/lib -lmmal_core -lmmal -l mmal_util -lvcos -lbcm_host -lopencv_core -lopencv_contrib -lopencv_imgproc -lopencv_highgui -lopencv_objdetect
