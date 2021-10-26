from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import cv2
from PIL import Image
import torch
import torchvision.models as models 
import pickle

#загружаем эмбединги и метки отдельно для мужчин и женщин
female_embed = np.loadtxt('female_embed.txt', dtype=float)
male_embed = np.loadtxt('male_embed.txt', dtype=float)
female_lab = np.loadtxt('female_lab.txt', dtype=str)
male_lab = np.loadtxt('male_lab.txt', dtype=str)

#создаём метки формата integer для дальнейшего использования в модели
lb_make = LabelEncoder()
names = np.unique(male_lab)
label_int = lb_make.fit_transform(names)
catigories_male = {}
for i in range(len(label_int)):
    catigories_male.update({names[i]: label_int[i]})

names = np.unique(female_lab)
label_int = lb_make.fit_transform(names)
catigories_female = {}
for i in range(len(label_int)):
    catigories_female.update({names[i]: label_int[i]})

female_label_int = [catigories_female[f'{x}'] for x in female_lab]
male_label_int = [catigories_male[f'{x}'] for x in male_lab]


#KNN для мужчин
X_train, X_test, y_train, y_test  = train_test_split(male_embed, male_label_int, test_size = 0.2, 
                                                     stratify = male_label_int, random_state = 1)

knn_men = KNeighborsClassifier()
knn_men.fit(X_train, y_train)

model_name = "knn_men.pkl"
with open(model_name, 'wb') as file:
    pickle.dump(knn_men, file)


#KNN для женщин
X_train, X_test, y_train, y_test  = train_test_split(female_embed, female_label_int, test_size = 0.2, 
                                                     stratify = female_label_int, random_state = 1)

knn_women = KNeighborsClassifier()
knn_women.fit(X_train, y_train)

model_name = "knn_women.pkl"
with open(model_name, 'wb') as file:
    pickle.dump(knn_women, file)


with open('catigories_male.pkl', 'wb') as file:
    pickle.dump(catigories_male, file)
with open('catigories_female.pkl', 'wb') as file:
    pickle.dump(catigories_female, file)