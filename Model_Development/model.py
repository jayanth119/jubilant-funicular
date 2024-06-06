import  librosa
import matplotlib.pyplot as plt
from google.colab import drive
import zipfile
import os
import pandas as pd
import IPython.display as ipd
import librosa.display
import numpy as np

# prompt: write program to extract zip folder in gdrive

# Mount your Google Drive
drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive")
# Get the path to the zip file
zip_path = "/content/drive/MyDrive/gender.zip"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content")

# Change directory to the extracted folder

# # Mount Google Drive
# drive.mount('/content/drive')

# # Path to your zip file
# zip_path = '/content/drive/MyDrive/gender'

# # Destination folder to extract the zip file
# extracted_folder = '/content/gender'


# # Check if the destination folder exists, if not, create it
# if not os.path.exists(extracted_folder):
#     os.makedirs(extracted_folder)
audiofile_male = '/content/gender/train/male69.wav'
audiofile_female =  '/content/gender/train/female100.wav'

data, sample_rate = librosa.load(audiofile_male)
print(data)
print(sample_rate)
# Plot the waveform
plt.figure(figsize=(14,5))
plt.plot(data)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
ipd.Audio(audiofile_male)


data, sample_rate = librosa.load(audiofile_female)
print(data)
print(sample_rate)
# Plot the waveform
plt.figure(figsize=(14,5))
plt.plot(data)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
ipd.Audio(audiofile_female)



def make_Dateframe(title, directory):
  cd = os.listdir(directory)
  print(cd)

  # Check if the file "prepocess.py" exists in the list
  if "prepocess.py" in cd:
    cd.remove("prepocess.py")

  # Check if the file "xm.py" exists in the list
  if "xm.py" in cd:
    cd.remove("xm.py")

  # Check if the file ".DS_Store" exists in the list
  if ".DS_Store" in cd:
    cd.remove(".DS_Store")

  label = []
  for i  in range(len(cd)):
    if cd[i][0]=="m":
      label.append(1)
    elif(cd[i][0]=="f"):
      label.append(0)
  return pd.DataFrame( {title:cd ,
                      "Target"  : label
                      })

train = make_Dateframe("train" ,"/content/gender/train")
train.head(10)

test =  make_Dateframe("train" ,"/content/gender/test")
test.head()



#implementation of feature extraction

def fe(file_name):
    # Check if the file has a valid audio file extension
    if not os.path.splitext(file_name)[1].lower() in ['.wav', '.mp3', '.ogg']:
        raise ValueError(f"Invalid audio file extension: {os.path.splitext(file_name)[1]}")

    # Extract features from the audio file
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features

xv  = fe('/content/gender/train/male69.wav')



train_path = '/content/gender/train'
test_path = '/content/gender/test'
trainfiles = os.listdir(train_path)
testfiles  = os.listdir(test_path)
  # Check if the file "prepocess.py" exists in the list
if "prepocess.py" in trainfiles:
    trainfiles.remove("prepocess.py")

  # Check if the file "xm.py" exists in the list
if "xm.py" in trainfiles :
    trainfiles.remove("xm.py")

  # Check if the file ".DS_Store" exists in the list
if ".DS_Store" in trainfiles:
    trainfiles.remove(".DS_Store")

if "prepocess.py" in testfiles:
    testfiles.remove("prepocess.py")

  # Check if the file "xm.py" exists in the list
if "xm.py" in testfiles :
    testfiles.remove("xm.py")

  # Check if the file ".DS_Store" exists in the list
if ".DS_Store" in testfiles:
    testfiles.remove(".DS_Store")


def madetest(files, test_path):
    data = []
    if len(files) > 0:
        for file in files:
            path = os.path.join(test_path, file)
            if not os.path.splitext(file)[1].lower() in ['.wav', '.mp3', '.ogg']:
                raise ValueError(f"Invalid audio file extension: {os.path.splitext(file)[1]}")
            label = 1 if file.startswith('male') else 0
            value = fe(path)  # Assuming fe() is defined elsewhere
            data.append((file, value, label))
        df = pd.DataFrame(data, columns=["name", "value", "target"]).sample(frac=1).reset_index(drop=True)
        return df
    
test_d = madetest(testfiles, test_path)
train_d = madetest(trainfiles , train_path)



test_d.head()

test_d.to_csv("Gender_test.csv")
train_d.to_csv("Gender_train.csv")



xtest = np.array(test_d['value'].tolist())
ytest = np.array(test_d['target'])
xtrain =np.array( train_d['value'].tolist())
ytrain = np.array(train_d['target'])


### Label Encoding
ytrain=np.array(pd.get_dummies(ytrain))
ytest = np.array(pd.get_dummies(ytest))


from keras.models import Sequential
from keras.layers import Dense  , Dropout , Flatten , Activation
from keras.layers import Conv2D , Convolution2D , MaxPooling2D , GlobalAveragePooling2D
from keras.optimizers import Adam

# from keras.utils import np_utils

from sklearn import metrics
rows = 40
columns = 173
channels = 1
num_labels = 2
filter_size = 2



from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(xtrain.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2, padding='same'))  # Adjusted pooling size and added padding
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling1D())

model.add(Dense(num_labels, activation='softmax'))



model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary
model.summary()




xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))





from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(xtrain, ytrain, batch_size=num_batch_size, epochs=num_epochs, validation_data=(xtest, ytest), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)




score = model.evaluate(xtrain, ytrain, verbose=0)
print("Training Accuracy: ", score[1])
print(xtest.shape)
print(xtrain.shape)
score = model.evaluate(xtest, ytest, verbose=0)
print("Testing Accuracy: ", score[1])



# prompt: predict using  .hdf5 file

from keras.models import load_model
from sklearn.metrics import confusion_matrix

import numpy as np

import  librosa
import matplotlib.pyplot as plt
from google.colab import drive
import zipfile
import os
import pandas as pd

def fe(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features

# Load the saved model
model = load_model('/content/GenderModelCNN.hdf5')
xv = fe("/content/WhatsApp Ptt 2024-05-31 at 21.36.05.ogg")
xv = xv.reshape((1, xv.shape[0], 1))

model.predict(xv)
# Make predictions on the test data


y_pred =  model.predict(xv)

# Convert the predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

print(y_pred)
print(y_pred_classes)