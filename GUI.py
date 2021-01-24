# Deep Learning (IC4263, ET4232) home assignment - system design
# Detecting pneumonia from chest x-ray images using convolutional neural networks
# B. Tech. IC-A 5 Aneesh Poduval
# B. Tech. IC-A 35 Johnathan Fernandes
# B. Tech. IC-A 15 Sarthak Chudgar
# B. Tech. ET-A 53 Siddhi Jadhav
# Submitted November 25, 2020

# Standalone GUI application

#%% Import packages
import cv2 # Read and process images
from tkinter import Label, Tk, Button, filedialog # Graphical User Interface
from PIL import Image, ImageTk # Image file handling
import numpy as np # Matrix handling
import tensorflow as tf # Neural networks
import os # Read system files

#%% Initialize GUI window

root = Tk() # Initialize main GUI window
root.title("Deep Learning based Pneumonia detection") # Set title of window
root.geometry("700x700") # Define window size

#%% Select image file from user

cwd = os.getcwd() # Store current working directory
root.filename =  filedialog.askopenfilename(initialdir = cwd,title = "Select image",filetypes = (("jpeg files","*.jpeg"),("png files","*.png"),("all files","*.*"))) # Accept file input from user

#%% Load model

best_model = tf.keras.models.load_model('checkpoints') # Import tensorflow model from loca disk

#%% Process input image

imgsize = 200 # Set image size to resize later
new_image = cv2.imread(root.filename,cv2.IMREAD_GRAYSCALE) # Read image as grayscale
imgtk = ImageTk.PhotoImage(image = Image.fromarray(new_image)) # Copy image for display in GUI later
new_resize = cv2.resize(new_image, (imgsize, imgsize)) # Resize image for better processing
new_data = np.array(new_resize) / 255 # convert to numpy array and normalize image
new_data = new_data.reshape(-1, imgsize, imgsize, 1) # Reshape to 4 dimensional tensor

#%% Run prediction model

testpred = (best_model.predict(new_data) > 0.5).astype("int32") # Convert probability to boolean

if testpred == 0:
    finalpred = "Healthy"
else:
    finalpred = "Pneumonia"

conf = best_model.predict(new_data)[0][0]*100  # Prediction probability * 100

#%% Define exit function

def quit():
    global root
    root.quit()
    root.destroy() # Closes GUI window

#%% Define GUI

text0 = Label(root, image=imgtk) # Display input image
text0.grid(column = 0, row = 0, columnspan = 2) # Define position

text1 = Label(root, text = "Prediction: ") # "Prediction" label
text1.grid(column = 0, row = 1,sticky='W')

text2 = Label(root, text = finalpred) # Predicted class
text2.grid(column = 1, row = 1,sticky='W')

text3 = Label(root, text = "Confidence: ") # "Confidence" label
text3.grid(column = 0, row = 2,sticky='W')

text4 = Label(root, text = conf) # Confidence
text4.grid(column = 1, row = 2,sticky='W')

text5 = Button(root, text="Quit", command=quit) # Exit button
text5.grid(column = 0, row = 4,sticky='W')

#%% GUI loop

root.mainloop() # Required by tkinter