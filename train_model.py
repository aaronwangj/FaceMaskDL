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

LEARNING_RATE = .0001
EPOCHS = 15
BATCH_SIZE = 32

#image location
DIRECTORY = r"C:\Users\aaronjw\Desktop\GitHub\FaceMaskDL\dataset"

#binary image categories
GROUPS = ["with_mask", "without_mask"]

print("Please wait: loading images...")

#image arrays stored in array
data = []
#corresponding labels stored in array
labels = []

#convert images into arrays, preprocess, put into arrays
for category in GROUPS:
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
extra = ImageDataGenerator(
    rotation_range=45,
    zoom_range=.15,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.15,
    horizontal_flip=True,
    fill_mode="nearest")

#Load MobileNetV2, default imagenets weights
firstModel = MobileNetV2(
    weights = "imagenet",
    include_top = False,
    input_tensor = Input(shape=(224, 224, 3)))

#second model on top of MobileNet
secondModel = firstModel.output
secondModel = AveragePooling2D(pool_size=(7,7))(secondModel)
secondModel = Flatten(name = "flatten")(secondModel)
secondModel = Dense(128, activation="relu")(secondModel)
secondModel = Dropout(0.5)(secondModel)
secondModel = Dense(2, activation="softmax")(secondModel)

#actual model, combining first model and secondModel
model = Model(inputs = firstModel.input, outputs = secondModel)

#freeze first model layers from learning during training
for layer in firstModel.layers:
    layer.trainable = False

#compile the model, adam optimizer
print("Compiling model...")
opt = Adam(lr = LEARNING_RATE, decay = LEARNING_RATE / EPOCHS)
model.compile(
    loss = "binary_crossentropy",
    optimizer = opt,
    metrics = ["accuracy"])

#train model
print("Training model...")
H = model.fit(
    extra.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs= EPOCHS)

#make predictions
print("Evaluating network...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)

#find corresponding labels
predIdxs = np.argmax(predIdxs, axis = 1)

#classification report
print(classification_report(testY.argmax(axis = 1), predIdxs, target_names = lb.classes_))

#save model
print("Saving model...")
model.save("detection.model", save_format="h5")

#plot accuracy and training loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label = "accuracy")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label = "val_accuracy")
plt.title = ("Training Loss and Accuracy")
plt.xlabel = ("Epoch #")
plt.ylabel = ("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")