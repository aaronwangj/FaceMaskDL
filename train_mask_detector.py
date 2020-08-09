from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = .0001
EPOCHS = 20
BS = 32

#image location
DIRECTORY = r"C:\Users\aaronjw\Desktop\GitHub\FaceMaskDL\dataset"

#binary image categories
CATEGORIES = ["with_mask", "without_mask"]

print("Please wait: loading images...")

#image arrays stored in array
data = []

#corresponding labels stored in array
labels = []

#convert images into arrays, preprocess, put into arrays
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size = (224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

#convert alphanumeric labels to one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#convert to np arrays
data = np.array(data, dtype = "float32")
labels = np.array(labels)

#splitting training/testing dataset
(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size = 0.20,
    stratify = labels,
    random_state = 42)

#increase dataset through augmentation
aug = ImageDataGenerator(
    rotation_range=45,
    zoom_range=.15,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.15,
    horizontal_flip=True,
    fill_mode="nearest")

#Load MobileNetV2, default imagenets weights
baseModel = MobileNetV2(
    weights = "imagenet",
    include_top = False,
    input_tensor = Input(shape=(224, 224, 3)))

#second model on top of MobileNet
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#actual model, combining baseModel and headModel
model = Model(inputs = baseModel.input, outputs = headModel)

#freeze baseModel layers from learning during training
for layer in baseModel.layers:
    layer.trainable = False

#compile the model, adam optimizer
print("Compiling model...")
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(
    loss = "binary_crossentropy",
    optimizer = opt,
    metrics = ["accuracy"])

#train model
print("Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs= EPOCHS)

#make predictions
print("Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

#find corresponding labels
predIdxs = np.argmax(predIdxs, axis = 1)

#classification report
print(classification_report(testY.argmax(axis = 1), predIdxs, target_names = lb.classes_))

#save model
print("Saving model...")
model.save("mask_detector.model", save_format="h5")

















