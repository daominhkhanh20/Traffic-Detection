import cv2
import numpy as np
import time
import argparse
from AssignLabel import predict_name
from PIL import Image
import os
import random

#hand comand line for run python
#when you run this file,you need must add 4 parameter: image,config,weight,classes
#ex: python3 main.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt


ap = argparse.ArgumentParser()

# ap.add_argument('-c', '--config', required=True,
#                 help = 'path to yolo config file')
# ap.add_argument('-w', '--weights', required=True,
#                 help = 'path to yolo pre-trained weights')
# ap.add_argument('-cl', '--classes', required=True,
#                 help = 'path to text file containing class names')

#ap.add_argument('-s','--size',required=True,help='size model image for CNN')
ap.add_argument('-v','--video',required=False)
ap.add_argument('-i', '--image',required=False,
               help = 'path to input image')
args = ap.parse_args()

classification=predict_name()

#load model, file configure model,name all of the class(80 class),random color for each class
def load_model():
    net=cv2.dnn.readNet('Yolo/yolov3.weights', 'Yolo/yolov3.cfg')
    classes=None
    with open('Yolo/yolov3.txt','r')as f:
        classes=[line.strip() for line in f.readlines()]

    layer_names=net.getLayerNames()
    output_layers=[layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]
    colors=np.random.randint(0,255,size=(43,3)).astype(float)#color for draw objects
    return net,classes,colors,output_layers
net,classes,colors,output_layers=load_model()

def load_image():
    image=cv2.imread(args.image)
    height,width=image.shape[:2]
    # if height >=1000 or width>=1000:
    #     image=cv2.resize(image,None,fx=0.5,fy=0.5)

    height,width=image.shape[:2]
    return image,height,width

def detect_objects(image,net,output_layers):
    '''
    +) To correctly predict the objects with deep neural network,we use function cv2.dnn.blobFromImage()
    +) net.forward(): return a nested list containing information about all the detected object
    which include x,y(coodinate the center object) and width,height for image,confidence score
    and score for all of the classes in file yolo.txt. The class with highest socre is predicted 
    for object
    '''
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(416,416),mean=(0,0,0),swapRB= True,crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs


def get_boxs(outputs,height,width):
    boxs=[]#list contain information for object(x_left_conner,y_left_conner,width,height)
    confidence_score=[]#confidence score
    class_ids=[]#index class in classes
    for output in outputs:
        for detect in output:
            scores=detect[5:]
            classid=np.argmax(scores)
            conf=scores[classid]
            if conf>0.5:
                x_center=int(detect[0]*width)
                y_center=int(detect[1]*height)
                w=int(detect[2]*width)
                h=int(detect[3]*height)
                x_left_conner=int(x_center-w/2)
                y_left_conner=int(y_center-h/2)
                boxs.append([x_left_conner,y_left_conner,w,h])
                class_ids.append(classid)
                confidence_score.append(float(conf))#need to convert to float,if not,error built-in appear
    return boxs,confidence_score,class_ids



def draw_labels(boxs,confidences,colors,classes,class_ids,image):
    confidence_threshold=0.5
    nms_threshold=0.4
    indexs=cv2.dnn.NMSBoxes(boxs,confidences,confidence_threshold,nms_threshold)
    for index in indexs:
        i=index[0]
        x,y,w,h= boxs[i]
        labelYolo=classes[class_ids[i]]+':{:.3f}'.format(confidences[i])
        color=colors[class_ids[i]]
    
        '''
        Note: In this cv2.rectangle(), parameter(x,y),(x+w,y+h) must be integer.
        Because of this error, I lost much time to find this bug. So anyone can read this line.
        Plsease keep this is mind
        '''
        z=5
        image_crop=image[y-z:y+h+z,x-z:x+w+z]
        name,index=classification.predict(image_crop)
        color=colors[index]
        cv2.rectangle(image, (x-z,y-z), (x+w+z,y+h+z),color, 2)

        label=str(name)+",id:"+str(index);
        cv2.putText(image, label, (x-50,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("Object detection",image)
    # os.getcwd();
    # os.chdir('Result')
    # name_file="Result{}.jpg".format(max(indexs)[0]*random.randint(1,900))
    # cv2.imwrite(name_file,image)
    # os.chdir("..")

def detect_object_image():
    image,height,width=load_image()
    outputs=detect_objects(image,net,output_layers)
    boxes,confidences,class_ids=get_boxs(outputs,height,width)
    draw_labels(boxes,confidences,colors,classes,class_ids,image)
    
    while True:
        key=cv2.waitKey(1)
        if key==27:#esc
            break

def detect_object_video():
    cap=cv2.VideoCapture(args.video)
    i=0
    while True:
        _,frame=cap.read()
        if(i%5==0):
            height,width,channel=frame.shape
            outputs=detect_objects(frame,net,output_layers)
            boxes,confidences,class_ids=get_boxs(outputs,height,width)
            draw_labels(boxes,confidences,colors,classes,class_ids,frame)
        
        i+=1
        key=cv2.waitKey(1)
        if key==27:
            break

    cap.release()

if __name__=="__main__":
    if args.image:
        detect_object_image()
    elif args.video:
        detect_object_video()