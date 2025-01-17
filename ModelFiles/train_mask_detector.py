import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import classification_report


DATADIR = r"C:\Users\user\dataset"
DATATYPE = ["with_mask","without_mask"]
data = []
labels = []

for dtype in DATATYPE:
    path = os.path.join(DATADIR, dtype)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(dtype)

#one hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(data, labels, test_size=0.2, stratify = labels , random_state = 42)

INIT_LR = 1e-4
EPOCHS = 3
BS = 32

# image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 20,
                        zoom_range = 0.15,
                        width_shift_range = 0.2,
                        height_shift_range = 0.2,
                        shear_range = 0.15,
                        horizontal_flip = True,
                        fill_mode = "nearest")

BaseModel = MobileNetV2(weights='imagenet', include_top=False,
                       input_tensor = Input(shape=(224,224,3)))

# fully connected layer
headModel = BaseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#actual model
model = Model(inputs=BaseModel.input, outputs=headModel)

for layer in BaseModel.layers:
    layer.trainable = False

#compile the model
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# training
H = model.fit(aug.flow(Xtrain,Ytrain, batch_size=BS),
             steps_per_epoch=len(Xtrain) // BS,
             validation_data=(Xtest, Ytest),
             validation_steps=len(Xtest) // BS,
             epochs = EPOCHS)

predIDXS = model.predict(Xtest, batch_size = BS)
predIDXS = np.argmax(predIDXS, axis=1)

# evaluation
print(classification_report(Ytest.argmax(axis=1), predIDXS, target_names=lb.classes_))

#save model
model.save("Face_Mask_detector.model", save_format="h5")

import matplotlib.pyplot as plt
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")