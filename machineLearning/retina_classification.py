import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

## Dataset from kaggle
##https://www.kaggle.com/datasets/paultimothymooney/kermany2018
##Comparing Normal with CNV

##ENABLING GPU PROCESSING TO INCREASE EXECUTIONAL PERFORMANCE
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

#IMPORTING THE DATA; IMAGES GET ASSIGNED TO A CLASS OF EITHER OF 0 FOR CNV OR 1 FOR Normal, BASED ON THE FOLDER THEY CAME FROM
#BATCH SIZE HAS BEEN SET AT 32, WHICH MEANS THE AI MODEL WITH TRAIN WITH 32 IMAGES AT A TIME
data = tf.keras.utils.image_dataset_from_directory('opht', batch_size = 32)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#SHOWING A SAMPLE OF THIS DATA, ALONG WITH THE 0 OR 1 CLASS
fig, ax = plt.subplots(ncols=10, figsize=(20,20))
for idx, img in enumerate(batch[0][:10]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

#SCALING THE DATA. 
#EACH RGB CHANNEL CAN HAVE A VALUE UP TO 255. BY DIVIDING BY 255, WE ARE SCALING THIS VALUE FROM A RANGE OF 0 TO 1, WHICH HELPS WITH MODEL OPTIMIZATION
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

#CREATING TRAINING, VALIDATION, AND TEST DATASET; THE DATASET IS CURRENTLY IN UNITS OF BATCHES, EACH WHICH HAS 32 AS DEFAULT
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
#CONFIRMING NUMBER OF BATCHES IN TRAINING SET
print(train_size)

#HERE WE ARE USING TAKE AND SKIP TO MAKE SURE THAT EACH DATASET HAS UNIQUE IMAGES
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#IMPORTING IN DEPENDENCIES
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
#CREATING THE MODEL
model = Sequential()

#PASSING IN ALL THESE LAYERS THAT BREAK DOWN THE IMAGE TO THE SEQUENTIAL CLASS
#THIS CONVOLUTION HAS 16 FILTERS, FILTER IS 3X3 PIXLES LONG, AND HAS A STRIDE OF 1 PIXEL.
#RELU TAKES THE OUTPUT FROM THE CONVOLUTIONAL LAYER AND MAKES NEGATIVE VALUES 0. THIS ESSENTIALLY AMPLIFIES THE SIGNAL. ANOTHER OPTION IS "SIGMOID"
#INPUT_SHAPE SHOULD MATCH THE DATASET; RECALL THAT THESE VALUES ARE DEFAULT BASED ON CODE ABOVE "data = tf.keras.utils.image_dataset_from_directory('data', batch_size = 10)"
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
#THIS TAKES MAX VALUE IN A 2X2 REGION (DEFAULT)
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#THIS TAKES ALL THE 2 DIMENTIONAL IMAGE, ACROSS ALL THE FILTERS (3RD DIMENTION), AND TURNS IT INTO A SINGLE DIMENTION --> 30*30*16=14400
model.add(Flatten())

#4096 NEURONS
model.add(Dense(4098, activation='relu'))
#SINGLE OUTPUT WITH A VALUE BETWEEN 0 AND 1
model.add(Dense(1, activation='sigmoid'))

#ADAM IS THE OPTIMIZER; AFTER EACH EPOCH, THE NN BECOMES MORE OPTIMIZED; BINARY CROSS-ENTROPY IS USED FOR BINARY CLASSIFICATION PROBLEMS
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

#####TRAINING#######
#CREATING LOG DIRECTORY
logdir='logs'
#CREATING CALL BACKS IN THE LOG DIRECTORY
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#FITTING THE DATA (TRAINING); EACH EPOCH IS A RUN THROUGH THE ENTIRE SET OF DATA; CAN MODIFY THIS TO ADDRESS OVERFITTING
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

#PLOTTING THE LOSS FROM THE HIST MODEL; IDEALLY BOTH OF THESE FIGURES SHOULD BE GOING DOWN TOGETHER; IF THEY DIVERGE, THIS IS AN INDICATION THAT YOU HAVE OVERFIT
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#PLOTTING ACCURACY FROM THE HIST MODEL
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

########EVALUATION##########
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

#THIS LOOPS THROUGH EACH BATCH AND DETERMINES THE PRECISION, RECALL, AND ACCURACY
for batch in test.as_numpy_iterator(): 
    X, y = batch
    #GENERATING PREDICTION
    yhat = model.predict(X)
    #COMPARING PREDICTION AGAINST ACTUAL VALUE
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

##SAVING THE MODEL FOR THE FUTURE
from tensorflow.keras.models import load_model
#SAVING IT IN A FOLDER CALLED "MODELS" WITH A NAME OF "IMAGE CLASSIFIER"
model.save(os.path.join('models','imageclassifier.h5'))





# #########LOADING THE MODEL############
# i =1
# while i <2:

#     from tensorflow.keras.models import load_model
#     new_model = load_model(os.path.join('models','imageclassifier.h5'))
#     ##IMPORTING IMAGE THAT THE MODEL HAS NEVER SEEN BEFORE
#     img = cv2.imread((input("type image name ")+".jpeg"))
    
#     plt.imshow(img)
#     # plt.show()
#     #THIS RESIZES THE IMAGE BEFORE IT GETS SENT INTO THE NN
#     resize1 = tf.image.resize(img, (256,256))
#     plt.imshow(resize1.numpy().astype(int))
#     # plt.show()

#     #EVALUATING THE MODEL AGAINST THE IMAGE AGAIN
#     yhat2 = new_model.predict(np.expand_dims(resize1/255, 0))

#     if yhat2 > 0.5: 
#         print(f'\nDIAGNOSIS: Normal')
#     else:
#         print(f'\nDIAGNOSIS: CNV')


#     print("PROBABILITY OF NORMAL DX:", " ", round(yhat2.flat[0], 2)*100, "%", "\n ", sep="")    
#     i+0
