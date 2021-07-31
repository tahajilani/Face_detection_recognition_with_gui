
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import os
from tkinter import *
from tkinter.ttk import *
import time

if not os.path.exists('images'):
    os.makedirs('images')


def set_GPU():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU check
    return device

def get_boxes(frame,device,max_box=False):
    if(max_box==True):
        mtcnn = MTCNN(margin=40, keep_all=True, device=device)
        boxes, _ = mtcnn.detect(frame)
        max=0
        index=0
        if boxes is not None:
            for i in range(boxes.shape[0]):
                x, y, w, h = boxes[i][:]
                area=(w-x)*(y-h)
                if(area>max):
                    max=area
                    index=i
            boxes=np.reshape(boxes[index],(1,4))
    else:
        mtcnn = MTCNN(margin=40, keep_all=True, device=device)
        boxes, _ = mtcnn.detect(frame)
    return boxes






def draw_box(frame,boxes,names):
    if boxes is not None:
        for i in range(boxes.shape[0]):
            if(i<len(names)):
                x,y,w,h=boxes[i][:]
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 1)
                cv2.circle(frame,(int(x), int(y)),1, (255, 0, 0), 5)
                cv2.circle(frame, (int(w), int(h)), 1, (255, 0, 0), 5)
                if(x<0):
                    x=0
                if(y<0):
                    y=0


                cv2.putText(frame, names[i], (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    return frame

def save_cropped(frame,boxes,token,count): #only save largest box

    if not os.path.exists('images'):
        os.mkdir('images')

    if boxes is not None:
        for i in range(boxes.shape[0]):
            x,y,w,h=boxes[i][:]

            if not os.path.exists('images' + '//' + str(token)):
                os.mkdir('images' + '//' + str(token))
                path = 'images' + '//' + str(token) + '/' + str(count) + '.jpg'
                cv2.imwrite(path, frame[int(y):int(h), int(x):int(w), :])
            else:
                path = 'images' + '//' + str(token) + '/' + str(count) + '.jpg'
                cv2.imwrite(path, frame[int(y):int(h), int(x):int(w), :])

    return True

def return_things(device):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    return mtcnn,resnet


def write_names(boxes,frame,device,mtcnn,resnet):
    name=[]
    if os.path.isfile('names.pt') and os.path.isfile('embeddings.pt'):
        names=torch.load('names.pt')
        embeddings=torch.load('embeddings.pt')
        names=torch.load('names.pt')
        embedding=torch.load('embeddings.pt')
        if boxes is not None:
            for i in range(boxes.shape[0]):
                x,y,w,h=boxes[i][:]
                new_img=frame[int(y):int(h), int(x):int(w), :]
                if new_img is not None and np.any(new_img):
                    new_img=cv2.resize(new_img,(512,512))
                    y_new = mtcnn(new_img)
                    if y_new is not None:
                        y_new = resnet(y_new.unsqueeze(0)).detach().cpu()
                        dists = np.array([[(e1 - e2).norm().item() for e2 in embeddings] for e1 in y_new])
                        if dists.min() <1.0:
                            name.append(names[dists.argmin()])
                        else:
                            name.append('')


                    else:
                        name.append('')
    else:
        name=[]
        if not boxes is None:
            for box in boxes:
                name.append('')
    return name

