from keras.utils import to_categorical
import tensorflow as tf
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,LeakyReLU,BatchNormalization
from keras.models import Sequential
from keras.callbacks import TensorBoard

PATH = r"Data\Training"
PATHS = os.listdir(PATH)

data_target = {re:int(i) for i,re in enumerate(PATHS)}

data  = []
target = []

def get_image():
    for catagory in PATHS:
        path = os.path.join(PATH,catagory)
        imgs = os.listdir(path)
        
        for i in imgs:
            img_path = os.path.join(path,i)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(50,50))

            target.append(data_target[catagory])
            yield img
      
for i in get_image():
    data.append(i)


data = np.array(data)/255
target = to_categorical(target)
print(target.shape)
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.2)

try:
    os.mkdir("Logs")
except:
    pass

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

def build_model():

    model = Sequential()

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    

    model.summary()
    
    return model


model = build_model()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics = ["accuracy"])
model.fit(train_data,train_target,epochs=50,validation_data=(test_data,test_target),callbacks=[tb_callback])
model.save("model.keras")