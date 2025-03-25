import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

model=keras.models.load_model("model.h5")
label=np.load("labels.npy")
data_size=0