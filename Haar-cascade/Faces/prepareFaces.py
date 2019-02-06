import sys
sys.path.append("C:\\OPENCV\\opencv\\build\\python\\2.7\\x86")
import cv2

import os.path
import os

SIZE = 100

cascade = cv2.cv.Load('c:/OPENCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
# cascade = cv2.cv.Load('c:/OPENCV/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml')
# cascade = cv2.CascadeClassifier('c:/OPENCV/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml')
storage = cv2.cv.CreateMemStorage()

if __name__ == "__main__":
    os.chdir("./src")
    
    BASE_PATH=os.getcwd()
    
    
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            os.chdir("..")
            try:
                os.stat("./" + subdirname)
            except:
                os.mkdir("./" + subdirname) 
            os.chdir("./" + subdirname)
            index = 0
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print "%s - %s" % (abs_path, subdirname)
                colorImage = cv2.cv.LoadImage(abs_path) # input image
                # grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
                # grayImage = cv2.cv.CloneImage(colorImage)
                # grayImage = cv2.cv.CreateImage ((colorImage.width,colorImage.height), 8, 1)
                # cv2.cv.CvtColor(colorImage, grayImage, cv2.cv.CV_BGR2GRAY)                
                detectedFace = cv2.cv.HaarDetectObjects(colorImage, cascade, storage,1.1,2,0,(150,150))
                detectedFace = cascade.detectMultiScale(colorImage,1,1)
                
                facesList = []
                # facesList.append(grayImage)
                # draw a green rectangle where the face is detected
                if detectedFace:
                    for face in detectedFace:
                        # x,y,w,h = [0][0], [0][1], [0][2], [0][3]
                        cv2.cv.Rectangle(colorImage,(face[0][0],face[0][1]), (face[0][0]+face[0][2],face[0][1]+face[0][3]), cv2.cv.RGB(155, 255, 25),2)
                        facesList.append(colorImage[face[0][1]:(face[0][1]+face[0][3]), face[0][0]:(face[0][0]+face[0][2])])
                        
                for i, face in enumerate(facesList):
                    thumbnail = cv2.cv.CreateMat(SIZE, SIZE, cv2.cv.CV_8UC1)
                    cv2.cv.Resize(face, thumbnail)
                    cv2.cv.SaveImage(subdirname + "_" + str(index) + ".jpg", thumbnail)
                        
                # cv2.namedWindow('Face Detection', cv2.CV_WINDOW_AUTOSIZE)
                # cv2.cv.ShowImage('Face Detection', grayImage) 
                # cv2.waitKey()
                index += 1
                
                
            label = label + 1
            
    cv2.destroyAllWindows()        # Closes displayed windows