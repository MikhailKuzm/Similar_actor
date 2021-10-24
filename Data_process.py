import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import torchvision.models as models 
import torch 
from torch import nn
from sklearn.preprocessing import LabelEncoder


# сделать цикл для таблицы: 1 столбик эмбединг, 2 - класс
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(512, 128)
#torch.save(resnet18.state_dict(), 'resnet18_weights.pth')
resnet18.load_state_dict(torch.load('resnet18_weights.pth'))
resnet18.eval()
resnet18 = resnet18.to(torch.float64) 

face_cascade=cv2.CascadeClassifier('.\\haarcascade_frontalface_default.xml')
genders = os.listdir('dataset')

embedings_frame = pd.DataFrame({'Embeding': [],
                                'Label': []})    

for gender in genders:
    path_to_gender = f'.\\dataset\\{gender}'
    actors_names = os.listdir(path_to_gender)
    for name in actors_names:
        print(name)
        path = f'.\\dataset\\{gender}\\{name}'
        images = os.listdir(path)
        if len(images) < 4:
            continue

        for image in images:
            path_to_image = path + '\\' + image
            image = Image.open(path_to_image)
            image = np.array(image)
            faces = face_cascade.detectMultiScale(image, 1.1, 4)
            if len(faces) != 1 or len(image.shape) != 3:
                continue


            for (x, y, w, h) in faces:    
                crop_img = image[y:y+h, x:x+w, :]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img = cv2.resize(crop_img, (200,150), interpolation = cv2.INTER_AREA)
                crop_img = np.transpose(crop_img, (2, 0, 1))

            face = np.expand_dims(crop_img, axis = 0)
            face = torch.tensor(face)
            face = face .to(dtype=torch.float64)
            with torch.no_grad():
                embeding = resnet18(face)

            embedings_frame = embedings_frame.append({'Embeding': embeding[0].detach().numpy(),
                                'Label': name},
                               ignore_index=True)


female_frame = embedings_frame[699:]
male_frame = embedings_frame[:699]

lb_make = LabelEncoder()
names = male_frame.Label.unique()
label_int = lb_make.fit_transform(names)
catigories_male = {}
for i in range(len(label_int)):
    catigories_male.update({names[i]: label_int[i]})

names = female_frame.Label.unique()
label_int = lb_make.fit_transform(names)
catigories_female = {}
for i in range(len(label_int)):
    catigories_female.update({names[i]: label_int[i]})
