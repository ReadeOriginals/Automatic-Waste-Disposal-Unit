#!/usr/bin/env python3

import cv2
import time
import numpy as np
from gpiozero import Servo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory()



# SERVO Initialisation
servo_1 = Servo(14, max_pulse_width= 2.3/1000, min_pulse_width= 0.5/1000, pin_factory=factory) # connected to GPIO 8
servo_2 = Servo(17, max_pulse_width= 2.3/1000, min_pulse_width= 0.5/1000, pin_factory=factory) # connected to GPIO 11
servo_3 = Servo(22, max_pulse_width= 2.3/1000, min_pulse_width= 1.4/1000, pin_factory=factory) # connected to GPIO 15
servo_4 = Servo(27, max_pulse_width= 2.3/1000, min_pulse_width= 1.4/1000, pin_factory=factory) # connected to GPIO 13


# Image Detection
classNames = []
classFile = "/home/speedyreadey/Desktop/Object_Detection_Files/coco.names" # find image library
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/speedyreadey/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/speedyreadey/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def findObject(img, thres, nms, draw=True, objects=[]): # function to call for image detection
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    itemList =[] # List of items being found
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                itemList.append([box,className])
                if (draw): # Draws green rectange aroun object
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,itemList # Returns item and position item was found


# Response to image detection
if __name__ == "__main__":
    fps = 1
    time_start = time.time()

    cap = cv2.VideoCapture(0) # Starts video capture
    cap.set(3,480)
    cap.set(4,480)
    cv2.waitKey(1)
    
    # Servo test (proves program has booted)
    servo_1.max()
    servo_3.min()
    servo_2.max()
    servo_4.min()
    sleep(1)
    servo_1.mid()
    servo_3.mid()
    servo_2.mid()
    servo_4.mid()
    
    while True:
        ret, frame = cap.read()
        time_current = time.time()

        if (time_current - time_start) > fps: # Limits frame rate to the image detection
            success, img = cap.read()
            result, itemList = findObject(img,0.4,0.2,objects=['bottle','banana','apple','orange', 'carrot', 'book', 'scissors', 'bowl', 'fork', 'spoon', 'plate']) # list of items to be found
            item_count = len(itemList)
            item_string = ' '.join(str(e) for e in itemList)
            print(item_string)
            #cv2.imshow("Output",img) # Uncomment if video feed is wanted to be outputted to a monitor
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(1) 

            if item_count > 0: # Servo responses to image detection
                if "bottle" in item_string:
                    print("Servo 1 working")
                    servo_1.max() # opens servo
                    sleep(3)
                    servo_1.mid() # closes servo
                    sleep(2)
                elif ("apple" or "banana" or "orange" or "carrot") in item_list and not "bowl" in item_string:
                    print("Servo 2 working")
                    servo_2.max() # opens servo
                    sleep(3)
                    servo_2.mid() # closes servo
                    sleep(2)
                elif ("bowl" or "spoon" or "fork" or "plate") and not "scissors" in item_string:
                    print("Servo 3 working")
                    servo_3.min() # opens servo
                    sleep(3)
                    servo_3.mid() # closes servo
                    sleep(2)
                elif "book" or "scissors" in item_string:
                    print("Servo 4 working")
                    servo_4.min() # opens servo
                    sleep(3)
                    servo_4.mid() # closes servo
                    sleep(2)
            time_start = time.time() # Resets time counter
            