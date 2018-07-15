from  keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
import numpy as np
import os
import cv2
import random

input_shape = (224, 224, 3)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    #Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    #Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    #Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    #Dense(4096, activation='relu'),
    Dense(128, activation='softmax')
])

def trainData_create():
    train_x = []
    train_y = []

    path = os.getcwd() + '/image'
    
    for i in range(0,20):
        dir = random.randint(1, 128)
        dir_path = os.path.join(path, str(dir))
        dir_list = os.listdir(dir_path)
        rd = random.randint(0, len(dir_list) - 1)
        imgpath = os.path.join(dir_path, dir_list[rd])
        img = cv2.imread(imgpath)
        x = cv2.resize(img, (224, 224))
        train_x.append(x)
        train_y.append(int(dir))

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_y1 = []
    for i in train_y:
        temp = np.zeros(128)
        temp[i-1] = 1
        train_y1.append(temp)
    train_y1 = np.asarray(train_y1)
    
    return train_x, train_y1

def run_model():
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    #print('Model compile complete.')
    #model.summary()
    model = load_model('vgg16_model_1000.h5')
    print("Model load complete.")
    for i in range(1, 10001):
        X_train, Y_train = trainData_create()
        print('Train freq:', i)
        model.fit(x=X_train, y=Y_train, epochs=1, batch_size=20)
        if i % 1000 == 0:
            model.save('vgg16_model_%d.h5' % (i))
            print("model save %d" % (i))
    print('Completely train model.')

if __name__ == "__main__":
    run_model()
