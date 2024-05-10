import keras
from PIL import Image
import numpy as np
import os

from matplotlib import pyplot as plt

Sequential = keras.models.Sequential
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense


DATASET_DIR = "dataset/"

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
           'w', 'x', 'y', 'z']


def convert_image_to_list(photo_name, letter):
    img = Image.open(DATASET_DIR + f"{letter}/{photo_name}").convert('L')
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 0:
                data[i][j] = 0
            else:
                data[i][j] = 1
    return data

def get_letter(photo_name):
    img = Image.open(photo_name).convert('L')
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 0:
                data[i][j] = 0
            else:
                data[i][j] = 1
    X_list = []
    X_list.append(data)

    X = np.array(X_list)

    model = keras.saving.load_model("model.keras")
    model.load_weights("model_weights.weights.h5")
    prediction = model.predict(X)
    print(letters[np.argmax(prediction)])

def train_model():
    X_list, Y_list = [], []

    for index, letter in enumerate(letters):
        for image in os.listdir(f"dataset/{letter}"):
            data = convert_image_to_list(image, letter)
            X_list.append(data)
            Y1_list = list(str("0" * 25))
            Y1_list = [int(i) for i in Y1_list]
            Y1_list.insert(index, 1)
            Y_list.append(Y1_list)


    X = np.array(X_list)
    Y = np.array(Y_list)

    print(X)
    print(Y)

    model = keras.Sequential([
        Flatten(input_shape=(32, 32, 1)),
        Dense(676, activation='relu'),
        Dense(26, activation='softmax')
    ])

    adam = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    his = model.fit(X, Y, batch_size=16, epochs=15)

    plt.plot(his.history['loss'])
    plt.savefig('graphic.png')

    model.save("model.keras")
    model.save_weights("model_weights.weights.h5")