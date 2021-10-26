import pickle
import cv2
from PIL import Image
import torch
import torchvision.models as models 
import numpy as np
from torch import nn

#загружаем архитектуру и веса resnet
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Linear(512, 128)
#torch.save(resnet18.state_dict(), 'resnet18_weights.pth')
resnet18.load_state_dict(torch.load('resnet18_weights.pth'))
resnet18 = resnet18.to(torch.float64) 
resnet18.eval()

#загружаем ранее оученные модели knn для мужчин и женщин
with open('knn_men.pkl', 'rb') as fid:
    knn_men = pickle.load(fid)
with open('knn_women.pkl', 'rb') as fid:
    knn_women = pickle.load(fid)

#загружаем словари для интерпритации результатов
with open('catigories_female.pkl', 'rb') as fid:
    catigories_female = pickle.load(fid)
with open('catigories_male.pkl', 'rb') as fid:
    catigories_male = pickle.load(fid)

#функция принимает путь к фото и пол человека на фото и возвращает имя наиболее похожего актёра
def predict(image_path, gender = 'man'):
    image = Image.open(image_path)
    image = np.array(image)
    face_cascade=cv2.CascadeClassifier('.\\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    if len(faces) != 1 or len(image.shape) != 3:
        print('More than 1 face or blac-white picture')
        return

    for (x, y, w, h) in faces:    
        crop_img = image[y:y+h, x:x+w, :]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_img = cv2.resize(crop_img, (200,150), interpolation = cv2.INTER_AREA)
        crop_img = np.transpose(crop_img, (2, 0, 1))

    face = np.expand_dims(crop_img, axis = 0)
    face = torch.tensor(face)
    face = face.to(dtype=torch.float64)
    
    with torch.no_grad():
        embeding = resnet18(face) 

    embeding = embeding.detach().numpy()
    if gender == 'man':
        y = knn_men.predict(embeding)
        new_ke_lis = list(catigories_male.keys())
        new_val = list(catigories_male.values())
        new_pos = new_val.index(y) # value from dictionary
        print("The most similar actor is",new_ke_lis[new_pos])

    else:
        y = knn_women.predict(embeding)
        new_ke_lis = list(catigories_female.keys())
        new_val = list(catigories_female.values())
        new_pos = new_val.index(y) # value from dictionary
        print("The most similar actor is",new_ke_lis[new_pos])

#проверям на существующей фотографии
predict('.\\dataset\\women\\АЛЁНА_САВАСТОВА\\2.jpeg', 'woman')  

    