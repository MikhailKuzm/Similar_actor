import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import torchvision.models as models 
import torch
from torch import nn



#загружаем  веса для нейронной сети resnet 18
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Linear(512, 128)
#torch.save(resnet18.state_dict(), 'resnet18_weights.pth')
resnet18.load_state_dict(torch.load('resnet18_weights.pth'))
resnet18.eval()
resnet18 = resnet18.to(torch.float64) 

#загружаем каскады Хаара для детекции лиц
face_cascade=cv2.CascadeClassifier('.\\haarcascade_frontalface_default.xml')
genders = os.listdir('dataset')

#пустая таблица для будущих эмбедингов и их классов
embedings = np.empty(128)
label = [] 

#загружаем фотографии актёров по одной
for gender in genders:
    path_to_gender = f'.\\dataset\\{gender}'
    actors_names = os.listdir(path_to_gender)
    for name in actors_names:
        print(name)
        path = f'.\\dataset\\{gender}\\{name}'
        images = os.listdir(path)
        if len(images) < 4:
            continue

        #если на фото более двух лиц или оно чёрно-белое то не трогаем фото
        for image in images:
            path_to_image = path + '\\' + image
            image = Image.open(path_to_image)
            image = np.array(image)
            faces = face_cascade.detectMultiScale(image, 1.1, 4)
            if len(faces) != 1 or len(image.shape) != 3:
                continue

            #вырезаем лицо на фото
            for (x, y, w, h) in faces:    
                crop_img = image[y:y+h, x:x+w, :]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img = cv2.resize(crop_img, (200,150), interpolation = cv2.INTER_AREA)
                crop_img = np.transpose(crop_img, (2, 0, 1))

            # приводим фото в формат, который ожидает нейронная сеть
            face = np.expand_dims(crop_img, axis = 0)
            face = torch.tensor(face)
            face = face .to(dtype=torch.float64)
            with torch.no_grad():
                embeding = resnet18(face)
            
            label.append(name)
            embedings = np.vstack((embedings, embeding.detach().numpy()))

#отделяем женщин от мужчин
female_embed = embedings[699:]
male_embed = embedings[:699]
female_lab = label[699:]
male_lab = label[:699]

np.savetxt('female_embed.txt', female_embed)
np.savetxt('male_embed.txt', male_embed)
np.savetxt('female_lab.txt', female_lab, fmt="%s")
np.savetxt('male_lab.txt', male_lab, fmt="%s")





