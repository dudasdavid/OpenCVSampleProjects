all: run

run: mobileNetSSD
	./mobileNetSSD
mobileNetSSD: mobileNetSSD.cpp
	g++ mobileNetSSD.cpp -o mobileNetSSD\
    -lraspicam -lraspicam_cv \
    -lmvnc -lpthread -lopencv_videoio -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_dnn \
    -L$(HOME)/git/wiringPi/wiringPi -lwiringPi
