from __future__ import print_function

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, concatenate
from keras.layers import Dense, Dropout, Embedding, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.layers import PReLU
from keras import regularizers
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sklearn as skplt
from mlxtend.plotting import plot_confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        name='confusion matrix.png',
                        cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    fig=plt.figure(1,figsize=(12,12))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        
    else:
        print('Confusion matrix without normalization')
    print('\n')   
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(name)
    fig.clear()


input_data_path = 'DATASET/' 
xpixels=150
ypixels=150
epochs =60
batch_size = 100
num_of_train_samples =2400
num_of_val_samples =2400
num_of_test_samples=3200
steps_per_epoch=num_of_train_samples//batch_size
validation_steps=num_of_val_samples//batch_size
test_steps=num_of_test_samples//100
        
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
X_train_gen = train_datagen.flow_from_directory(input_data_path + 'train/',
            class_mode='categorical',
            target_size=(xpixels,ypixels),
            color_mode='grayscale',
            shuffle=True)
X_val_gen = validation_datagen.flow_from_directory(input_data_path + 'validation/',
            class_mode='categorical',
            target_size=(xpixels,ypixels),
            color_mode='grayscale',
            shuffle=True)
X_test_gen = test_datagen.flow_from_directory(input_data_path + 'test/',
            class_mode='categorical',
            target_size=(xpixels,ypixels),
            color_mode='grayscale',
            batch_size=100,
            shuffle=False)

model = Sequential()

model.add(Conv2D(8, kernel_size=(3, 3), input_shape=(xpixels,  ypixels, 1)))
#model.add(PReLU(alpha_regularizer=regularizers.l2(0.05)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(4, kernel_size=(3, 3)))
model.add(PReLU(alpha_regularizer=regularizers.l2(0.05)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(4, kernel_size=(3, 3)))
model.add(PReLU(alpha_regularizer=regularizers.l2(0.05)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(8))
model.add(PReLU(alpha_regularizer=regularizers.l2(0.05)))


model.add(Dense(X_val_gen.num_classes))
model.add(Activation('softmax'))
opt=keras.optimizers.SGD(momentum=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history=model.fit_generator(
            X_train_gen,
            verbose=1,
            steps_per_epoch=steps_per_epoch, #22 should typically be equal to the number of unique samples of your dataset divided by the batch size
            epochs=epochs,
            validation_data=X_val_gen,
            validation_steps=validation_steps)


print('Model Summary:\n')

model.summary()

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

for i in range(2):
    if(i==0):
        fig = plt.figure()
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.savefig('accuracy.png'.format(i))
        plt.close()
    if(i==1):
        fig = plt.figure()
        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Training and validation loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('loss.png'.format(i))
        plt.close()

Y_pred = model.predict_generator(X_test_gen,test_steps)
predicted_classes = np.argmax(Y_pred, axis=1)
true_classes = X_test_gen.classes
class_labels = list(X_test_gen.class_indices.keys())   
target_names = ['NC', 'ABF', 'VSG','VSW','WD',
                'UVF','OVF','FT']
#print('Confusion Matrix')
cnf_mat=confusion_matrix(true_classes,predicted_classes)

plot_confusion_matrix(cm=cnf_mat, classes=target_names, title='Confusion Matrix',name='cnf.png')

print('Classification Report')

print(classification_report(X_test_gen.classes, predicted_classes, target_names=target_names))

