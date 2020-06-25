import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#%% device config Extra
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

#%% 
def read_images(path, numberofimg):
    array = np.zeros([numberofimg, 64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "/" + img
        img = Image.open(img_path, mode="r")
        data = np.asarray(img,dtype = "uint8")
        data = data.flatten()
        array[i,:] = data
        i += 1
    return array
#%% read train negative
train_neg_path = r"/Users/fuatsezer/Desktop/Görüntü İşleme/LSIFIR/Classification/Train/neg" 
numberoftrain_neg_img = 43390
train_neg_array = read_images(train_neg_path,numberoftrain_neg_img)
#%%
x_train_negative_tensor = torch.from_numpy(train_neg_array)
print(x_train_negative_tensor.size())
#%%
y_train_negative_tensor = torch.zeros(numberoftrain_neg_img, dtype = torch.long)
print(y_train_negative_tensor.size())

#%% read train positive
train_pos_path = r"/Users/fuatsezer/Desktop/Görüntü İşleme/LSIFIR/Classification/Train/pos" 
numberoftrain_pos_img = 10208
train_pos_array = read_images(train_pos_path,numberoftrain_pos_img)
#%%
x_train_positive_tensor = torch.from_numpy(train_pos_array)
print(x_train_positive_tensor.size())
#%%
y_train_positive_tensor = torch.zeros(numberoftrain_pos_img, dtype = torch.long)
print(y_train_positive_tensor.size())

#%% concat train
x_train = torch.cat((x_train_negative_tensor,x_train_positive_tensor),0)
y_train = torch.cat((y_train_negative_tensor,y_train_positive_tensor),0)
#%%
print(x_train.size())
print(y_train.size())

#%% read train negative
test_neg_path = r"/Users/fuatsezer/Desktop/Görüntü İşleme/LSIFIR/Classification/Test/neg" 
numberoftest_neg_img = 22050
test_neg_array = read_images(test_neg_path,numberoftest_neg_img)
#%%
x_test_negative_tensor = torch.from_numpy(test_neg_array[:20855,:])
print(x_test_negative_tensor.size())
#%%
y_test_negative_tensor = torch.zeros(20855, dtype = torch.long)
print(y_test_negative_tensor.size())

#%% read train positive
test_pos_path = r"/Users/fuatsezer/Desktop/Görüntü İşleme/LSIFIR/Classification/Test/pos" 
numberoftest_pos_img = 5944
test_pos_array = read_images(test_pos_path,numberoftest_pos_img)
#%%
x_test_positive_tensor = torch.from_numpy(test_pos_array)
print(x_test_positive_tensor.size())
#%%
y_test_positive_tensor = torch.zeros(numberoftest_pos_img, dtype = torch.long)
print(y_test_positive_tensor.size())

#%% concat train
x_test = torch.cat((x_test_negative_tensor,x_test_positive_tensor),0)
y_test = torch.cat((y_test_negative_tensor,y_test_positive_tensor),0)
#%%
print(x_test.size())
print(y_test.size())

#%% Visualize 
plt.imshow(x_train[10000,:].reshape(64,32),cmap="gray")

#%% CNN
# Hyperparameter
num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,10,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,16,5)
        
        self.fc1 = nn.Linear(16*13*5,500)
        self.fc1 = nn.Linear(520,130)
        self.fc1 = nn.Linear(130,num_classes)
        
        
        
    
    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool1(F.relu(self.conv2(x)))
        
        x.view(-1*16*13*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
#%%   
import torch.utils.data

train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train , batch_size=batch_size,shuffle=True)
#%%
test = torch.utils.data.TensorDataset(x_test,y_test)
testloader = torch.utils.data.DataLoader(test , batch_size=batch_size,shuffle=True)

#%%
net = Net()










        

