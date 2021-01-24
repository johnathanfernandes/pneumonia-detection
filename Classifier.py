# Deep Learning (IC4263, ET4232) home assignment - system design
# Detecting pneumonia from chest x-ray images using convolutional neural networks
# B. Tech. IC-A 5 Aneesh Poduval
# B. Tech. IC-A 35 Johnathan Fernandes
# B. Tech. IC-A 15 Sarthak Chudgar
# B. Tech. ET-A 53 Siddhi Jadhav
# Submitted November 25, 2020

# Model creation

#%% Imports

import os # Read system files
import cv2 # Read and process images
import numpy as np # Matrix handling
import matplotlib.pyplot as plt # Data visualization
from keras.preprocessing.image import ImageDataGenerator # Data augmentation
import tensorflow as tf # Neural networks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # Model callbacks for training
from sklearn.metrics import confusion_matrix # Performance metrics

#%% Import and label data

folders = ["NORMAL", "PNEUMONIA"] # Define list of classes
imgsize = 200 # Set image size to resize later

def getdata(datapath):
    data = [] # Initialize empty list for data
    for folder in folders:
        path = os.path.join(datapath,folder) # Read path of folder
        label = folders.index(folder) # Get label from folder name
        for img in os.listdir(path):
            imgarray = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE) # Read images into memory
            resize = cv2.resize(imgarray, (imgsize, imgsize)) # Resize to specified size
            data.append([resize, label]) # Attach label to image
    return np.array(data)

#%%

train = getdata("data/train")
#print("train set initialized")
test = getdata("data/test")
#print("test set initialized")
valid = getdata("data/val")
#print("validation set initialized")

#%% Check class balance

def classcount(dataset):
    healthy = sick = 0
    for i in dataset:
        if(i[1] == 0):
            healthy +=1
        else:
            sick += 1
    return [healthy, sick]

#%% Visualize class balance

labels = ['Training', 'Testing', 'Validation'] # x axis labels

healthy = [classcount(train)[0],classcount(test)[0],classcount(valid)[0]] # Count of healthy samples
sick = [classcount(train)[1],classcount(test)[1],classcount(valid)[1]] # Count of pneumonia samples

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, healthy, width, label='Healthy')
rects2 = ax.bar(x + width/2, sick, width, label='Pneumonia')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Set')
ax.set_title('Number of samples by class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects): # Attach a text label above each bar in *rects*, displaying its height
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

#%% Define lists for all sets

xtrain = []
ytrain = []

for feature, label in train:
    xtrain.append(feature)
    ytrain.append(label)

xtest = []
ytest = []

for feature, label in test:
    xtest.append(feature)
    ytest.append(label)

xvalid = []
yvalid = []

for feature, label in valid:
    xvalid.append(feature)
    yvalid.append(label)

#%% Normalize values for better performance

xtrain = np.array(xtrain) / 255
xtest = np.array(xtest) / 255
xvalid = np.array(xvalid) / 255

#%% Resize data to 4-dimensional tensors

xtrain = xtrain.reshape(-1, imgsize, imgsize, 1)
xtest = xtest.reshape(-1, imgsize, imgsize, 1)
xvalid = xvalid.reshape(-1, imgsize, imgsize, 1)

ytrain = np.array(ytrain)
yvalid = np.array(yvalid)
ytest = np.array(ytest)

#%% Data augmentation

datagen = ImageDataGenerator(featurewise_center=False, # Set input mean to 0 over the dataset, feature-wise
                             samplewise_center=False, #  Set each sample mean to 0
                             featurewise_std_normalization=False, # Divide inputs by std of the dataset, feature-wise
                             samplewise_std_normalization=False, # Divide each input by its std
                             zca_whitening=False, # Apply ZCA whitening
                             zca_epsilon=1e-6, # epsilon for ZCA whitening
                             rotation_range=120, # Degree range for random rotations
                             width_shift_range=0.5, # fraction of total width
                             height_shift_range=0.5, # fraction of total height
                             brightness_range=None, # Range for picking a brightness shift value from
                             shear_range=0., # Shear angle in counter-clockwise direction in degrees
                             zoom_range=0.8, # Range for random zoom
                             channel_shift_range=0., # Range for random channel shifts
                             fill_mode='nearest', # Points outside the boundaries of the input are filled according to the given mode
                             cval=0., # Value used for points outside the boundaries when fill_mode = "constant"
                             horizontal_flip=False, # Randomly flip inputs horizontally
                             vertical_flip=False, # Randomly flip inputs vertically
                             rescale=None, # rescaling factor
                             preprocessing_function=None, # function that will be applied on each input
                             data_format=None, # Image data format
                             validation_split=0.0, # Fraction of images reserved for validation
                             dtype=None) # Dtype to use for the generated arrays

datagen.fit(xtrain)

#%% Define model

model = tf.keras.models.Sequential() # groups a linear stack of layers into a  model

model.add(tf.keras.layers.Conv2D(16, # Dimensionality of the output space
                                 (5,5), # Filter size
                                 activation='relu', # Activation function
                                 strides = 1, # Stride length
                                 input_shape=(imgsize,imgsize,1))) # Input image shape
model.add(tf.keras.layers.MaxPooling2D(2,2)) # Max pooling layer with 2x2 filter size
model.add(tf.keras.layers.BatchNormalization()) # Batch normalization layer

model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2)) # Dropout layer with 20% dropout rate
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(256,(1,1),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten()) # Flattens the input
model.add(tf.keras.layers.Dense(512,activation='relu')) # densely-connected NN layer
model.add(tf.keras.layers.Dropout(0.7))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1,activation='sigmoid')) # Output layer with sigmoid activation function

model.compile(optimizer='adam', # Configures the model for training using "Adam" optimizer
              loss='binary_crossentropy', # Loss function
              metrics=['accuracy']) # Performance metric
#model.summary() # View model summary

#%% Define callbacks
checkpoint = ModelCheckpoint("checkpoints", # Directory
                             monitor='accuracy', # Performance metric to monitor
                             verbose=1, # Print update messages
                             save_best_only=True, # Save only best performing model
                             save_weights_only=False, # Save only weights from model
                             mode='max', # Criteria to replace saved model
                             save_freq='epoch') # Frequency to save model

earlystop = EarlyStopping(monitor='accuracy',
                          min_delta=0, # Minimum change in the monitored quantity
                          patience=7, # Number of epochs with no improvement
                          verbose=1,
                          mode='max',
                          baseline=None, # Baseline value for the monitored quantity
                          restore_best_weights=True) # restore model weights from the epoch with the best value of the monitored quantity

lrreduction = ReduceLROnPlateau(monitor='accuracy',
                                factor=0.1, # new lr = lr * factor.
                                patience = 4, # number of epochs with no improvement
                                verbose=1,
                                mode='max',
                                min_delta=1e-4, # threshold for measuring the new optimum
                                cooldown=0, # number of epochs to wait before resuming normal operation after lr has been reduced
                                min_lr=0.000001) # lower bound on the learning rate

callbacks = [checkpoint, earlystop, lrreduction]
#%% Train model

history = model.fit(datagen.flow(xtrain,ytrain, batch_size = 32), # Train model on augmented dataset
                    epochs = 100, # Number of epochs
                    validation_data = datagen.flow(xvalid, yvalid), # Validate on augmented data
                    callbacks = callbacks) # Define callbacks

#%% Load model
best_model = tf.keras.models.load_model('checkpoints')

#%% Predict on testing data

mloss = best_model.evaluate(xtest,ytest)[0] # Model loss
macc = best_model.evaluate(xtest,ytest)[1] # Model accuracy

pred = (best_model.predict(xtest) > 0.5).astype("int32") # Convert probability to boolean
pred = pred.reshape(1,-1)[0]

cm = confusion_matrix(ytest,pred) # Generate confusion matrix

#%% Predict on test image

new_image = cv2.imread("p1.jpeg",cv2.IMREAD_GRAYSCALE) # Read image as grayscale
new_resize = cv2.resize(new_image, (imgsize, imgsize)) # Resize image for better processing
new_data = np.array(new_resize) / 255 # convert to numpy array and normalize image
new_data = new_data.reshape(-1, imgsize, imgsize, 1) # Reshape to 4 dimensional tensor
testpred = (best_model.predict(new_data) > 0.5).astype("int32") # Convert probability to boolean
if testpred == 0:
    finalpred = "Healthy"
else:
    finalpred = "Pneumonia"

