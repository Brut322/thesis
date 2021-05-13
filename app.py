import numpy as np # linear algebra
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from PIL import Image as im
from tensorflow.keras import layers,models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


train='fruits-360/Training/'
test='fruits-360/Test/'

X_train_img=np.array(glob(train+'/**/*.jpg',recursive=True))
X_test_img=np.array(glob(test+'/**/*.jpg',recursive=True))

y_train=[]
for x in X_train_img:
    y_train.append(x.split('/')[3])
y_train=pd.DataFrame(data=y_train,columns=['test'])

print(len(X_train_img))

y_test=[]
for y in X_test_img:
    y_test.append(y.split('/')[3])
y_test=pd.DataFrame(y_test,columns=['test'])

k=pd.concat([y_train.test,y_test.test])
z=LabelEncoder()
p=z.fit_transform(k)
classifier=[[a,b] for a,b in zip(pd.DataFrame(p)[0].unique(),k.unique())]
y_train=p[:len(y_train)]
y_test=p[len(y_train):]

print(cv2.imread(X_train_img[1100]).shape)

X_train=[]
X_test=[]

for i in range(len(X_train_img)):
    X_train.append(cv2.imread(X_train_img[i]))

X_train=np.array(X_train)

for j in range(len(X_test_img)):
    X_test.append(cv2.imread(X_test_img[j]))

X_test=np.array(X_test)

data_augmentation=models.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical',input_shape=(100,100,3)),
    layers.experimental.preprocessing.RandomContrast(0.85),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.2)
])

CNN=models.Sequential([
    data_augmentation,
    layers.Conv2D(filters=20,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=40,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=80,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
    layers.Dense(512,activation='relu'),
    layers.Dense(106,activation='softmax')
])

model_json = CNN.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

CNN.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=CNN.fit(X_train,y_train,epochs=25)

print(history)

print(CNN.evaluate(X_test,y_test))

CNN.save('thesis_network.h5')

y_pred=CNN.predict(X_test)
print(y_pred[:10])

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

y_pred_classes=[np.argmax(element) for element in y_pred]
print("Classfication report\n",classification_report(y_test,y_pred_classes))
