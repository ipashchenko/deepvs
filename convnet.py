import os
from keras import models
from keras import layers
from keras import optimizers
from data_loading import get_data


data_dir = "/home/ilya/Dropbox/papers/ogle2/data"
var_fname_file = os.path.join(data_dir, "LMC_SC20_vars_lcfnames.txt")
const_fname_file = os.path.join(data_dir, "LMC_SC20_const_lcfnames.txt")
images, labels = get_data(var_fname_file, const_fname_file, data_dir,
                          const_number=168)
images = images.astype('float32') / 255
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels =\
    train_test_split(images, labels, test_size=0.1, stratify=labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(23, 24, 1)))
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=50, batch_size=64)