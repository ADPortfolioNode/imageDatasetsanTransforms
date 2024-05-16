import requests
import tarfile 

import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd


import os
from torchvision import transforms

# Directory to save the files
directory = "resources/data"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# URLs for the files
img_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/img.tar.gz"
index_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/index.csv"
 


# Download and save the img.tar.gz file
response = requests.get(img_url, stream=True)
with open(os.path.join(directory, "img.tar.gz"), "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

# Extract the img.tar.gz file
with tarfile.open(os.path.join(directory, "img.tar.gz"), "r:gz") as tar:
    tar.extractall(path=directory)

# Download and save the index.csv file
response = requests.get(index_url)
with open(os.path.join(directory, "index.csv"), "wb") as file:
    file.write(response.content)
    
    
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])
    
# the libraries we are going to use for this lab. The torch.manual_seed() is for forcing the random function to give the same number every time we try to recompile it.

# These are the libraries will be used for this lab.


#AUXILIARY FUNCTIONS
print("AUXILIARY FUNCTIONS")
directory='resources/data'
csv_file='index.csv'
csv_path=os.path.join(directory,csv_file)
data_name = pd.read_csv('resources/data/index.csv')
data_name.head()

# Get the value on location row 0, column 1 (Notice that index starts at 0)
#rember this dataset has only 100 samples to make the download faster  
print('File name:', data_name.iloc[0, 1])

# Get the value on location row 0, column 0 (Notice that index starts at 0.)

print('y:', data_name.iloc[0, 0])

# Print out the file name and the class number of the element on row 1 (the second row)

print('File name:', data_name.iloc[1, 1])
print('class or y:', data_name.iloc[1, 0])

# Print out the total number of rows in traing dataset

print('The number of rows: ', data_name.shape[0])

#LOAD IMAGE
print("LOAD IMAGE")

# Combine the directory path with file name

image_name =data_name.iloc[1, 1]
print("image_name ", image_name)

image_path=os.path.join(directory,image_name)
print("image path ",image_path)

# Plot the second training image

image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[1, 0])
plt.show()

# Plot the 20th image

image_name = data_name.iloc[19, 1]
image_path=os.path.join(directory,image_name)
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[19, 0])
plt.show()

# Create a dataset class
print("Create a dataset class")

# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
    
# Create the dataset objects

dataset = Dataset(csv_file=csv_file, data_dir=directory)

image=dataset[0][0]
y=dataset[0][1]

plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()


#plot the second image
image=dataset[9][0]
y=dataset[9][1]

plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()        


#torch image transforms
print("torch image transforms")

# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=croptensor_data_transform )
print("The shape of the first element tensor: ", dataset[0][0].shape)


# Plot the first element in the dataset

show_data(dataset[0],shape = (20, 20))

# Plot the second element in the dataset

show_data(dataset[1],shape = (20, 20))

# Construct the compose. Apply it on MNIST dataset. Plot the image out.

fliptensor_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1),transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform )
show_data(dataset[1])

# Practice: Combine vertical flip, horizontal flip and convert to tensor as a compose. 
# Apply the compose on image. Then plot the image
print("my code")
fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1), transforms.RandomVerticalFlip(p = 1 ), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform )
show_data(dataset[2]) 

print("my code ", dataset[1][0].shape)
print("my code ", dataset[2][0].shape)
print("my code ", dataset[1][1])
print("my code ", dataset[2][1])
# Type your code here

print("their code")

my_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1),transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform )
show_data(dataset[1])
print("their code ", dataset[1][0].shape)
print("their code ", dataset[2][0].shape)
print("their code ", dataset[1][1])
print("their code ", dataset[2][1])


print(">>>>>>>>>>>>>>>>End of Line<<<<<<<<<<<<<<<<<<<<<")