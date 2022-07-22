
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import csv
#import pandas as pd
tf.__version__
keras.__version__

print("Hello World!")
#(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X=np.load("D:\\ML_project\\project_debarthi\\X.npy")
Y=np.load("D:\\ML_project\\project_debarthi\\y.npy")
#X= pd.read_csv (r'C:\\Users\\hp\\Desktop\\project_debarthi\\tensor\\X.csv')
#y=pd.read_csv (r'C:\\Users\\hp\\Desktop\\project_debarthi\\tensor\\Y.csv')
# split into train test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.05)

#X_train_full.shape
#X_train_full.dtype
X_valid, X_train = X_train_full[:1000] / 255., X_train_full[1000:] / 255.#to normalize so that optimizers like SGD wors well
y_valid, y_train = y_train_full[:1000], y_train_full[1000:]
X_test = X_test / 255.
#plt.imshow(X_train_full[1], cmap="binary")
#plt.axis('off')
#plt.show()

#plt.imshow(y_train_full[1], cmap="binary")
#plt.axis('off')
#plt.show()
#y_train


from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='selu', padding="SAME")#how to arrange the convolution
model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[64, 64, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),#creating  image vector
    
    
    keras.layers.Dense(units=64, activation='selu',kernel_initializer="lecun_normal"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=128, activation='selu',kernel_initializer="lecun_normal"),
    keras.layers.Dropout(0.5),#increases accuracy
    keras.layers.Dense(units=256, activation='selu',kernel_initializer="lecun_normal"),
    keras.layers.Dropout(0.5),#increases accuracy
    keras.layers.Dense(units=64, activation='selu',kernel_initializer="lecun_normal"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax')#softmax to select particular class from many
])
# model = keras.models.Sequential([
#     keras.layers.Dense(4096, activation="relu"),
#     keras.layers.Dense(3500, activation="relu"),
#     keras.layers.Dense(2300, activation="relu"),
#     keras.layers.Dense(1800, activation="relu"),
#     keras.layers.Dense(1300, activation="relu"),
#     keras.layers.Dense(600, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(5, activation="softmax")
# ])

input_shape = X_train.shape  
model.build(input_shape)
model.compile(loss="categorical_crossentropy",#image is categorical data
             optimizer="sgd",#to reduce error in each iteration
              metrics=["accuracy"])
model.layers
model.summary()

# print("line 51")
# model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("D:\\ML_project\\project_debarthi\\mymodel3.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=80,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
model = keras.models.load_model("D:\\ML_project\\project_debarthi\\mymodel3.h5") # rollback to best model
