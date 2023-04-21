import torch
import numpy as np

#----------------------------------------------------
#Tensor Initialization
#Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data:  {x_data} \n");

#from a Numpy array
np_array = np.array(data);
x_np = torch.from_numpy(np_array);
print(f"x_np: {x_np} \n");

# return scale values of 1 in tensors, keep the same size
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n");

#random tensor and also change the data type
x_rand = torch.rand_like(x_data, dtype=torch.float32);
print(f"Random Tensor: \n {x_rand} \n");

#With random or constant values
shape = (2,3)
rand_tensor = torch.rand(shape);
ones_tensor = torch.ones(shape);
zeros_tensor = torch.zeros(shape);

print(f"Random Tensor from Shape: \n {rand_tensor} \n");
print(f"One Tensor from Shape: \n {ones_tensor} \n");
print(f"Zero Tensor from Shape: \n {zeros_tensor} \n");


#----------------------------------------------------
#Tensor Attributes
tensor = torch.rand(3, 4);
print(f"Shape of Tensor: {tensor.shape}");
print(f"Datatype of Tensor: {tensor.dtype}");
print(f"Device Tensor is stored on: {tensor.device} \n");


#----------------------------------------------------
#Tensor Operations
if torch.cuda.is_available():
    tensor = tensor.to("cuda");
    print(f"Device tensor is stored on: {tensor.device} \n");
    
#numpy-like indexing and slicing
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(f"Sliced indexed Tensor: {tensor}\n");

#Joining tensors, concatenate, dim starting from 0
t1 = torch.cat([tensor, tensor, tensor],dim=1)
print(f"catenated tensors {t1}");

#multiplying tensors element-wise
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n");

#Alternative syntax
print(f"tensor*tensor \n {tensor*tensor} \n");

#matrix multiplication between two tensors, matmul or @
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n");

#Alternative syntax
print(f"tensor@tensor.T \n {tensor@tensor.T} \n");

#----------------------------------------------------
#Bridge with Numpy
t = torch.ones(5)
print(f"t: {t}");
n = t.numpy();
print(f"n: {n} \n");

#A change in the tensor reflects in the Numpy array as well
t.add_(1);   #in-place adding
print(f"t: {t}");
print(f"n: {n} \n");

#Numpy array to Tensor
n = np.ones(5);
t = torch.from_numpy(n);

#changes in the Numpy array reflects in the tensor as well
np.add(n, 1, out=n);
print(f"t: {t}");
print(f"n: {n} \n");

