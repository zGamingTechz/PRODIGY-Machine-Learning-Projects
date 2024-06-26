import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kaggle_input_directory = '/kaggle/input/food-101/food-101/food-101/images/'
classes = os.listdir(kaggle_input_directory)

PATH = '/kaggle/input/food-101/food-101/food-101/images/'

transform = transforms.Compose(
    [transforms.Resize((64,64)),
     transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) 


trainset = torchvision.datasets.ImageFolder(PATH, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=0, shuffle=True)

testset = torchvision.datasets.ImageFolder(PATH, transform=transform)

num_samples = 100
indices = torch.randperm(len(testset))[:num_samples]

subset_data = Subset(testset, indices)

testloader = DataLoader(subset_data, batch_size=64, num_workers=0, shuffle=True)

def image_shower(images, labels, n=4):
    plt.figure(figsize=(12, 12))
    for i, image in enumerate(images[:n]):
        plt.subplot(n, n, i + 1)
        image = image/ 2 + 0.5
        plt.imshow(image.numpy().transpose((1, 2, 0)).squeeze())
    print("Real Labels: ", ' '.join('%5s' % classes[label] for label in labels[:n]))

images, labels = next(iter(trainloader))
image_shower(images, labels)

model = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():
    param.require = False

model.fc = nn.Linear(2048, len(classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 9
model.to(device)

for epoch in range(epochs):
    running_loss = 0.0
  
    for i, data in tqdm(enumerate(trainloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print ("Epoch {} - Training loss: {} ".format(epoch, running_loss/len(trainloader)))

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
      
        _, predicted = torch.max(outputs.data, 1)
      
        total += labels.size(0)
      
        correct += (predicted == labels).sum().item()
print("Accuracy: %d" %(100 * correct/total))

images, labels = next(iter(testloader))
image_shower(images, labels)

outputs = model(images.to(device))

predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join("%5s" %classes[predict] for predict in predicted[:4]))

torch.save(model.state_dict(), '/kaggle/working/model2.pt')

