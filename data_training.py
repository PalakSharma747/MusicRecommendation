import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Initialize variables
is_init = False
label = []
dictionary = {}
c = 0

# Load .npy files
for i in os.listdir():
    if i.endswith('.npy'):
        data = np.load(i)

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Reshape to match feature count

        # Debugging: Print the shape of each loaded file
        print(f"Loaded {i}, shape: {data.shape}")

        size = data.shape[0]  # Number of samples

        # Skip incompatible files
        if is_init and X.shape[1] != data.shape[1]:
            print(f"Skipping {i}: Dimension mismatch (Expected {X.shape[1]}, got {data.shape[1]})")
            continue  

        # Initialize X and y
        if not is_init:
            is_init = True
            X = data
            y = np.full((size, 1), i.split(".")[0])
        else:
            X = np.concatenate((X, data), axis=0)  # Concatenate along samples (rows)
            y = np.concatenate((y, np.full((size, 1), i.split(".")[0])), axis=0)

        label.append(i.split(".")[0])
        dictionary[i.split(".")[0]] = c
        c += 1

# Debug: Check dataset sizes
print(f"Final X shape: {X.shape}, y shape: {y.shape}")

# Convert labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

y = np.array(y, dtype="int32")
y = keras.utils.to_categorical(y)

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Model definition
input_layer = keras.layers.Input(shape=(X.shape[1],))
hidden_1 = keras.layers.Dense(512, activation="relu")(input_layer)
hidden_2 = keras.layers.Dense(256, activation="relu")(hidden_1)
output_layer = keras.layers.Dense(y.shape[1], activation="softmax")(hidden_2)
model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=50)

# Save model and labels
model.save("model.h5")
np.save('labels.npy', np.array(label))

# import os
# import numpy as np
# import cv2
# from tensorflow import keras
# # from tensorflow.keras.utils import to_categorical
# #  from keras.layers import Input , Dense
# # from keras.models import Model
# is_init =False
# size = -1
# label =[]
# dictionary ={}
# c = 0
# for i in os.listdir():   #we will search for file in the same directory
    
#      if i.split(".")[-1] == 'npy':
#         if not(is_init):
#              is_init = True
#              X = np.load(i)
#              size = X.shape[0]
#              #y = np.array([i.split(".")[0]]*size).reshape(-1,1)
#              y = np.full((size, 1), i.split(".")[0])  # Corrected label array size

            
#         else:
#             X=np.concatenate((X ,np.load(i)))
#             y=np.concatenate((y ,np.array([i.split(".")[0]]*size).reshape (-1,1)))
            
#         label.append(i.split(".")[0])    
#         dictionary[i.split(".")[0]] = c
#         c =c+1
# print(dictionary) 
# print(label)           
# # print(X.shape) 
# # print(y.shape)      
# # print(X)
# # print(y)     
# for i in range (y.shape[0]):
#     y[i,0] =dictionary[y[i , 0]]
  
# y = np.array(y, dtype="int32")
# # print(y)  

# y =keras.utils.to_categorical(y)
# # print(y)
# # X_new =X.copy()
# # y_new=y.copy()
# # counter=0
# # cnt=np.arange(X.shape[0])  #provides a list 
# # np.random.shuffle(cnt)

# # for i in cnt:
# #     X_new[counter] = X[i]
# #     y_new[counter] = y[i]
# #     counter=counter+1
# # # print(y)
# # ip=keras.layers.Input(shape=(X.shape[1]))

# ip = keras.layers.Input(shape=(X.shape[1],))  # Add a comma



# m=keras.layers.Dense(512 , activation ="relu")(ip)
# n=keras.layers.Dense(256 , activation ="relu")(m)

# op=keras.layers.Dense(y.shape[1], activation='softmax')(m)
# model =keras.models.Model(inputs =ip , outputs=op)

# model.compile(optimizer='rmsprop' , loss='categorical_crossentropy',metrics=['acc'])
# model.fit(X , y ,epochs=50)
# # model.save("model.h5")
# # np.save('labels.npy' , np.array(label))