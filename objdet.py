import numpy as np
import pandas as pd
import torch
import cv2
from ultralytics import YOLO
#
cap=cv2.VideoCapture(0)
model1 =YOLO("/Users/navtegh/Downloads/best.pt")
# path='/usr/local/lib/python3.9/dist-packages/yolov5/yolov5s-int8.tflite'
#count=0

# model = torch.hub.load('/usr/local/lib/python3.9/dist-packages/yolov5', 'custom', path,source='local')
# model = model1.predict(source="0",show=TRUE, conf=0.5)
# b=model.names[2] = 'car'


size=416

count=0
car=0
toy=0
bottle=0


color=(0,0,255)

cy1=250
offset=30

while True:
    ret,img=cap.read()

    count += 1
    # if count % 2 != 0:
    #     continue
    img=cv2.resize(img,(1000,700))
    # cv2.line(img,(300,500),(300,0),(0,0,255),2)
    results = model1.predict(source=img, conf=0.7)
    xyxy=0
    classid=5
    if(results[0]):
        for x in results[0]:
            xyxy =x.boxes.xyxy.cpu().numpy().astype(int)
            classid = x.boxes.cls.cpu().numpy().astype(int)
            print(xyxy[0][0])
            print(classid)
        # if(results[0]):
        #     x=results[0].numpy()
        #     print(x)
        x1 = xyxy[0][0]
        y1 = xyxy[0][1]
        x2 = xyxy[0][2]
        y2 = xyxy[0][3]
        rectx1, recty1 = ((x1 + x2) / 2), ((y1 + y2) / 2)
        rectcenter = int(rectx1), int(recty1)
        cx = rectcenter[0]
        cy = rectcenter[1]
        if(classid==0):
            cv2.rectangle(img,(xyxy[0][0],xyxy[0][1]),(xyxy[0][2],xyxy[0][3]),(0,0,255),2)
            cv2.putText(img, 'bottle', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if cx < (500 + offset) and cx > (500 - offset):
                bottle += 1
                cv2.line(img, (500, 700), (500, 0), (0, 255, 0), 2)
        elif (classid == 1):
            cv2.rectangle(img, (xyxy[0][0], xyxy[0][1]), (xyxy[0][2], xyxy[0][3]), (0, 255, 0), 2)
            cv2.putText(img, 'car', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if cx < (500 + offset) and cx > (500 - offset):
                car += 1
                cv2.line(img, (500, 700), (500, 0), (0, 255, 0), 2)
        elif (classid == 2):
            cv2.rectangle(img, (xyxy[0][0], xyxy[0][1]), (xyxy[0][2], xyxy[0][3]), (255, 0, 0), 2)
            cv2.putText(img, 'toy', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if cx < (500 + offset) and cx > (500 - offset):
                toy += 1
                cv2.line(img, (500, 700), (500, 0), (0, 255, 0), 2)


        cv2.circle(img,(cx,cy),3,(0,255,0),-1)
    cv2.putText(img, 'bottle = ' + str(bottle), (800, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(img, 'car = ' + str(car), (800, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(img, 'toy = ' + str(toy), (800, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)





    # a=results
    cv2.imshow("IMG",img)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()