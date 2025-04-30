import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical 
from keras.layers import Input, Dense 
from keras.models import Model
from sklearn.metrics import classification_report

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not(is_init):
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c+1

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1

ip = Input(shape=(X.shape[1]))
m = Dense(128, activation="tanh")(ip)#128
m = Dense(64, activation="tanh")(m)#64
op = Dense(y.shape[1], activation="softmax")(m) 
model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X_new, y_new, epochs=10, validation_split=0.2)#epochs=80

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_new, y_new)
print("Validation Accuracy:", accuracy)

# Predict classes for validation set
y_pred = np.argmax(model.predict(X_new), axis=1)

# Convert one-hot encoded labels back to original labels
y_true = np.argmax(y_new, axis=1)

# Calculate precision, recall, and F1 score
report = classification_report(y_true, y_pred, target_names=label)
print("Classification Report:")
print(report)

# # Save the trained model
# model.save("modelH.h5")
# np.save("labelH.npy", np.array(label))
# # Save the trained model

model.save("modelTH.h5")
np.save("labelTrH.npy", np.array(label))
