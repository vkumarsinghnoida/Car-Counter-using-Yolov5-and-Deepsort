import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture('video.mp4')
whT = 320
 
#### LOAD MODEL
## Coco Names
classesFile = "custom.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files
modelConfiguration = "yolov4-tiny-detector.cfg"
modelWeights = "yolov4-tiny-detector2.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)
 
def findObjects(img, confThreshold, nmsThreshold):
    
    frame_rate_calc = 1
    freq = cv.getTickFrequency()
    t1 = cv.getTickCount()
    

 
    
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                if True:
                    w,h = int(det[2]*wT) , int(det[3]*hT)
                    x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3] 
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 3)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    t2 = cv.getTickCount()
    time1 = (t2-t1)/freq
    fps= 1/time1
    #cv.putText(img,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv.LINE_AA)
    
    return len(indices), fps

''' 
while True:
    
    frame_rate_calc = 1
    freq = cv.getTickFrequency()
    t1 = cv.getTickCount()
    
    success, img = cap.read()
 
    
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)
    time.sleep(0)
    
    t2 = cv.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    cv.putText(img,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv.LINE_AA)

 
    cv.imshow('Image', img)
    
    if cv.waitKey(1) == 13:
        break
    
    
   ''' 
cv.destroyAllWindows()   