#autograd section
# forward propagation and backward propagation

#CPU Version only - Use Case
import torch
from torchvision.models import resnet18, ResNet18_Weights

#downloand and prepare initial data
model = resnet18(weights=ResNet18_Weights.DEFAULT);
data = torch.rand(1, 3, 64, 64);
labels = torch.rand(1, 1000);

prediction = model(data);  #forward pass

#backwardpass
loss = (prediction-labels).sum();
loss.backward();

#load the optimizer for SGD Performing
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9);
optim.step()  #gradient descent


#Differentiations in autograd
a = torch.tensor([2., 3.], requires_grad=True);
b = torch.tensor([6., 4.], requires_grad=True);

Q = 3*a**3 - b **2;

#The output tensor of an operation will require gradients even if only a single input tensor has requires_grad=True.
x = torch.rand(5, 5);
y = torch.rand(5, 5);
z = torch.rand((5, 5), requires_grad=True);

a = x+y;
b = y+z;
print(f"Does 'a' require gradients?: {a.requires_grad}");
print(f"Dose 'b' require gradients?: {b.requires_grad}");


# frozen parameters for partial update
model = resnet18(weights = ResNet18_Weights.DEFAULT);

#Freeze all the parameters in the network
model.fc = torch.nn.Linear(512, 10);  #except the fc layer

#optimize only the classifier
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9);

