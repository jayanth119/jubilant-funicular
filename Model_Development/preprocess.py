import os
import shutil
import random


def create_train_test_split(src_dir, train_dir, test_dir, train_ratio=0.8):
    # Ensure the destination directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of files
    files = os.listdir(src_dir)
    random.shuffle(files)

    # Split files into train and test
    split_index = int(len(files) * train_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(train_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(test_dir, file))

    print("Train-test split complete.")

# Define source and destination directories
source_directory = 'females'
train_directory = 'train'
test_directory = 'test'

cd = os.listdir()
cd.remove( 'prepocess.py')
cd.remove('.DS_Store')
for  i in range(len(cd)):
    os.rename(cd[i], f"female{i}.wav")
print("completed >>> ")

# Create train-test split
create_train_test_split(source_directory, train_directory, test_directory)

import os 

cd = os.listdir()
m = 0 
f = 0 
for i in range(len(cd)):
    if(cd[i][0]=="m"):
        m+=1 
    elif (cd[i][0]=="f") :
        f+=1

print("male" , m)
print("female" , f)
# test 
# male 443
# female 443
# test 
# male 1829
# female 1829

