#review neural networks
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__();
        
        #1 input image channel, 6 output channels, 5x5 square convolution
        #kernel
        self.conv1 = nn.Conv2d(1, 6, 5);
        self.conv2 = nn.Conv2d(6, 16, 5);
        
        #an affine operation
        self.fc1 = nn.Linear(16*5*5, 120);  #5*5 from image dimension
        self.fc2 = nn.Linear(120, 84);
        self.fc3 = nn.Linear(84, 10);
        
    def forward(self, x):
        #define the forwarding pipeline with activation functions
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # (2,2) window for pooling
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = torch.flatten(x, 1); # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));
        x = self.fc3(x);
        return x;
        
        
net = Net();
print(f"Neural Network: {net}\n");

#learnable parameters
params = list(net.parameters());
print(len(params));
print(params[0].size())  #conv1's weight

#try with random inputs

input = torch.randn(1, 1, 32, 32);
out = net(input);  #perform the forward computing
print(f"output after net {out}");

#fed random gradients for test
net.zero_grad();
out.backward(torch.randn(1, 10));

#torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
#For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
#If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

#Calculate the loss - MSELoss
output = net(input);
target = torch.randn(10);
target = target.view(1, -1);
criterion = nn.MSELoss();

loss = criterion(output, target);
print(f"Loss: {loss.grad_fn}"); #MSELoss
print(f"Linear Function: {loss.grad_fn.next_functions[0][0]}");
print(f"Relu Function: {loss.grad_fn.next_functions[0][0].next_functions[0][0]}");

#back propagate the error
net.zero_grad();
print(f"conv1.bias grad before backward")
print(net.conv1.bias.grad)

loss.backward();

print("conv1.bias.grad after backward");
print(net.conv1.bias.grad);

#update weight using in-place subtraction
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data* learning_rate)

#update weight with optimizer
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01);

# do this in the training loop
optimizer.zero_grad();   #clear the grad accumulation during the backprop
output = net(input)
loss = criterion(output, target);
loss.backward();
optimizer.step() #Dose the update


