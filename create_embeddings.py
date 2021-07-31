
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import cv2


def create_embeddings():

    workers = 0 if os.name == 'nt' else 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))


    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    pictures=[]

    directory=os.listdir('images_cropped')
    for i in range(len(directory)):
        count = 0
        flag = False
        while(flag==False):
            path='images_cropped'+'/'+str(directory[i])+'/'+str(count)+'.jpg'
            pic=cv2.imread(path)
            print(count)
            if pic is not None:
                tens=mtcnn(pic)
                if tens is not None:
                    pictures.append(tens)
                    flag=True
                    break
                    count=0
                    print(directory[i])
                    print('here')
                else:
                    count=count+1
            else:
                print(directory[i])
                count=count+1


    print(pictures)

    embeddings=[]

    for x in pictures:
        embeddings.append(resnet(x.unsqueeze(0)).detach().cpu())

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=directory, index=directory))
    torch.save(embeddings, 'embeddings.pt')  # Of course, you can also save it in a file.
    torch.save(directory, 'names.pt')


