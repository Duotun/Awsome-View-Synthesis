import torch
#torchvision is mainly responsible for the dataset loader
import torchvision
import torchvision.transforms as transforms

#use CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');
print(f"device: {device}");


#download and configure dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
);

batch_size = 4;
trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#for windows may only support 0 num_worker
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform);

testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0);

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#check imported images
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + 0.5;
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))); #rows, cols, color channels
    plt.show();
    
#get some random training images and show
dataiter = iter(trainLoader)
images, labels = next(dataiter)

#imshow(torchvision.utils.make_grid(images));
#print labels with 5 width
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)));  
print(f"image size {images.shape}");

#define a convolutional naeural network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5);
        self.fc1  = nn.Linear(16*5*5, 120);
        self.fc2 = nn.Linear(120, 84);
        self.fc3 = nn.Linear(84, 10);
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)));
        x = self.pool(F.relu(self.conv2(x)));
        x = torch.flatten(x, 1) #flatten all the dimensions except batch
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));
        x = self.fc3(x);
        return x 
    
net = Net();
net.to(device);

#define a loss function and optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9);

#Train the network
#loop over the dataset multiple times
'''
for epoch in range(2):
    running_loss = 0.0;
    for i, data in enumerate(trainLoader, 0):
        #get the inputs; 
        inputs, labels = data[0].to(device), data[1].to(device);
        
        #zero the parameter gradients
        optimizer.zero_grad();
        
        #forward + backward + optimize
        outputs = net(inputs);
        loss = criterion(outputs, labels);
        loss.backward();
        optimizer.step();
        
        #print statistics
        running_loss += loss.item();
        if i%2000 == 1999:
            #print every 2000 mini-batches
            print(f'[{epoch+1}, {i+1: 5d}] loss: {running_loss/2000.0:.3f}');
            running_loss = 0.0
print('Finished Training');     

''' 
#quickly save the model
Path = './Cifar_net.pth'
#torch.save(net.state_dict(), Path);
 

#Test the network on the test data
dataiter = iter(testLoader);
images, labels = next(dataiter)

#print images
#imshow(torchvision.utils.make_grid(images));
#print('GroundTruth: ', ''.join(f'{classes[labels[j]]: 5s}' for j in range(4)));

#load the state from file back to net
net = Net();
net.load_state_dict(torch.load(Path));

outputs = net(images);
_, predicted = torch.max(outputs, 1);

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)));

#check how correctness of the network
correct = 0;
total = 0;

#since we are not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data;
        #calculate outputs by running images
        outputs = net(images);
        
        #the class with the highest energy to be chosen
        _, predicted = torch.max(outputs.data, 1);
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f'Accuracy of the network on the 10000 test images: {100 * correct //total} %');

#check which class the trained model performs well and not well
#use dict
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testLoader:
        images, labels =data;
        outputs = net(images)
        _, predictions = torch.max(outputs, 1);
        #collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] +=1
            total_pred[classes[label]]+=1
            
#print accuracy for each label
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count)/ total_pred[classname];
    print(f'Accuracy for class: {classname:5s} is {accuracy:1f} %');



