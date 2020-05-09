import keras
import os, shutil
from keras.datasets import cifar100
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image


base_dir = os.path.join(os.getcwd(),"data")


train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')


train_bear_dir = os.path.join(train_dir, 'bear')
train_butterfly_dir = os.path.join(train_dir, 'butterfly')
train_girl_dir = os.path.join(train_dir, 'girl')
train_keyboard_dir = os.path.join(train_dir, 'keyboard')
train_table_dir = os.path.join(train_dir, 'table')
train_trout_dir = os.path.join(train_dir, 'trout')


test_bear_dir = os.path.join(test_dir, 'bear')
test_butterfly_dir = os.path.join(test_dir, 'butterfly')
test_girl_dir = os.path.join(test_dir, 'girl')
test_keyboard_dir = os.path.join(test_dir, 'keyboard')
test_table_dir = os.path.join(test_dir, 'table')
test_trout_dir = os.path.join(test_dir, 'trout')


validation_bear_dir = os.path.join(validation_dir, 'bear')
validation_butterfly_dir = os.path.join(validation_dir, 'butterfly')
validation_girl_dir = os.path.join(validation_dir, 'girl')
validation_keyboard_dir = os.path.join(validation_dir, 'keyboard')
validation_table_dir = os.path.join(validation_dir, 'table')
validation_trout_dir = os.path.join(validation_dir, 'trout')


#ağı oluşturma
model=models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#model.add(layers.Dropout((0.5)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        #bu hedef dizindir.
        train_dir,
        #Tüm resimler 150*150 boyutuna getirilir.
        target_size=(150, 150),
        batch_size=20,
        #loss fonksiyonunda categorical_crossentropy kullandığımız için
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#eğitim
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

#veri zenginleştirme
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,)


epochs = range(len(acc))
#epochs=range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


plt.show()







