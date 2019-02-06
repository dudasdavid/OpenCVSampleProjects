all: run

run: ObjectTracker
	./ObjectTracker
ObjectTracker: ObjectTracker.cpp
	g++ ObjectTracker.cpp -o ObjectTracker\
    -lraspicam -lraspicam_cv \
    -lmvnc -lpthread -lopencv_videoio -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_dnn -lopencv_tracking \
    -L$(HOME)/git/wiringPi/wiringPi -lwiringPi
