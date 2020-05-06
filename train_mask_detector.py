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
import argparse
import os



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())


# learning rate number of epochs and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


# getting the path of images
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# preprocessing images and getting their labels 
for imagePath in imagePaths:
	# since the directory is /dataset/with_labels/images.png
	label = imagePath.split(os.path.sep)[-2]

	# loading the images and preprocessing them
	image = load_img(imagePath, target_size = (224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)


# performing one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partitioning to training and test set
trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 42)

# training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range = 20,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.15,
	horizontal_flip = True, 
	fill_mode = "nearest"
)

# loading the MobileNetVet network
basemodel = MobileNetV2(weights="imagenet", include_top=False, input_tensor = Input(shape=(224, 224, 3)))

model = basemodel.output
model = AveragePooling2D(pool_size=(7, 7))(model)
model = Flatten(name="flatten")(model)
model = Dense(128, activation="relu")(model)
model = Dropout(0.5)(model)
model = Dense(2, activation="softmax")(model)

# placing the new model on top of the base model 
model = Model(inputs=basemodel.inputs, outputs = model)

# looping over layers in base models from updating during the first training process
for layer in basemodel.layers:
	layer.trainable = False

# compiling the model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# training the nets
H = model.fit(
	aug.flow(trainX, trainY, batch_size = BS),
	steps_per_epoch = len(trainX) // BS,
	validation_data = (testX, testY),
	validation_steps = len(testX) // BS,
	epochs = EPOCHS
)

# predictions on Test set
predictions = model.predict(testX, batch_size = BS)

# finding the index of the largest probability
predictions = np.argmax(predictions, axis = 1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predictions,
	target_names=lb.classes_))

# saving the model
model.save(args["model"], save_format = "h5")

# Plotting the training loss and accuaracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])