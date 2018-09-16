import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from sklearn.metrics import confusion_matrix
from data_loading import get_data
from utils import smooth_curve
import matplotlib.pyplot as plt


data_dir = "/home/ilya/Dropbox/papers/ogle2/data"
var_fname_file = os.path.join(data_dir, "LMC_SC20_vars_lcfnames.txt")
const_fname_file = os.path.join(data_dir, "LMC_SC20_const_lcfnames.txt")
# images, labels = get_data(var_fname_file, const_fname_file, data_dir,
#                           const_number=1000)
# images = images.astype('float32') / 255
images = np.load("images_1000.npy")
labels = np.load("labels_1000.npy")
# np.save("images_1000", images)
# np.save("labels_1000", labels)
n_positive = np.count_nonzero(labels)
n_negative = len(labels) - n_positive
imbalance_ratio = float(n_negative)/n_positive
print("Imabalance = {}".format(imbalance_ratio))
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels =\
    train_test_split(images, labels, test_size=0.33, stratify=labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(23, 24, 1)))
model.add(layers.Dropout(0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks_list = [callbacks.EarlyStopping(monitor="val_loss", patience=100),
                  callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=50)]
history = model.fit(train_images, train_labels, epochs=1000, batch_size=64,
                    validation_data=(test_images, test_labels),
                    class_weight={1: imbalance_ratio, 0: 1.0},
                    callbacks=callbacks_list)
                    # )
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test loss = {}; test acc = {}".format(test_loss, test_acc))

pred_labels = model.predict(test_images, batch_size=64)
pred_labels = [1. if y_ > 0.5 else 0. for y_ in pred_labels]

CMs = list()
CMs.append(confusion_matrix(test_labels, pred_labels))
CM = np.sum(CMs, axis=0)
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print("TP = {}".format(TP))
print("FP = {}".format(FP))
print("FN = {}".format(FN))
f1 = 2.*TP/(2.*TP+FP+FN)
print("F1 = {}".format(f1))

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'o', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel("epoch")
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'o', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel("epoch")
plt.legend()
plt.show()