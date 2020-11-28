import os 
import cv2
import numpy as np
import tqdm

group=[[0,1,2,3,4,5,7,8,9,10,15,16],#circular,white ground with red border
        [12,13,14,17],#other
        [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],#triangle,white ground with read border
        [32,41,42],#white ground with black inside
        [6,33,34,35,36,37,38,39,40]]#circular,blue ground
classes={}

def assign_label_for_each_group():
    for i in range(43):
        for index,j in enumerate(group):
            if i in j:
                classes[i]=index


def yolo_annotation(file_name,annotation,height,width):
    os.chdir("..")
    os.chdir("annotations")

    label=classes[annotation[4]]
    annotation[2]-=annotation[0]#w
    annotation[3]-=annotation[1]#h
    annotation[0]+=annotation[2]/2 #x
    annotation[1]+=annotation[3]/2 #y
    annotation[0]/=width
    annotation[1]/=height
    annotation[2]/=width
    annotation[3]/=height

    new_line=str(label)+" "+str(annotation[0])+" "+str(annotation[1])+" "+str(annotation[2])+" "+str(annotation[3])
    with open(file_name+'.txt',"a")as file_out:
        file_out.write(new_line)
        file_out.write("\n")
        file_out.close()
    os.chdir("..")


def get_index(s):
    if s[2]!='0':
        return 2
    elif s[3]!='0':
        return 3
    return 4

if __name__=="__main__":
    assign_label_for_each_group()
    #print(classes)
    path=os.getcwd()
    f=open('gt.txt')
    os.chdir('image')
    file_temp=""
    temp=[i for i in range(900)]
    j=0
    while True:
        line=f.readline()
        if not line:
            break
        data=line.split(';')
        file,annotation=data[0],np.array([float(data[1]),float(data[2]),float(data[3]),float(data[4]),int(data[5])])
        file_name=str.split(file,".")[0]
        if j!=0 and file_name!=file_temp:
            index1=get_index(file_temp)
            idnex2=get_index(file_name)
            x1=int(file_temp[index1:])
            x2=int(file_name[idnex2:])
            for i in range(x1+1,x2,1):
                cmd='mv /home/daominhkhanh/Downloads/Convert/image/'+'0'*(5-len(str(i)))+str(i)+'.ppm /home/daominhkhanh/Downloads/Convert/test'
                os.system(cmd)
            temp.remove(x2)
            file_temp=file_name
        else:
            file_temp=file_name
        j+=1

        height,width=cv2.imread(file_name+'.ppm').shape[:2]
        yolo_annotation(file_name,annotation,height,width)
        os.chdir("image")
    
print(temp)