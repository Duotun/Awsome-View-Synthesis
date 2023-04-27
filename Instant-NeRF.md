<!-- omit in toc -->
# :cyclone: Instant NGP with Taichi
This document records the basic understanding and implementations for Instant NGP with [Taichi](https://github.com/taichi-dev/taichi-nerfs).

- Instant NGP: 
  1) "Occupancy Grid" for accelerations of NeRF's training
  2) "Hash Encoder" for accelerations of NeRF's convergence rate 
![Hash Table](images/hash_table.png)


- NeRF Pipeline:
![NeRF Pipeline](./images/nerf_struct.png)
![AutoGrad and AutoDiff](./images/autograd.png)
Generally, there are several components in the NeRF including "Positional Encoding", "The Radiance Field Function Approximator", "Differentiable Volume Renderer", "Stratified Sampling", and "Hierarchical Volume Sampling".

- Ray Marching for Rendering Results:
   - Early Stop with High Opacity
![Ray Marching](./images/ray_marching.png)
![Early Stop](./images/early_stop.png)


---
## Volumetric Rendering
Here attached an implementation and tutorials of Volumetric Rendering in [Shadertoy](https://disigns.wordpress.com/portfolio/shadertoy-glsl-demos/) for practice and reviews.

Volume rendering is a technique for visualizing sampled functions of a colored semi transparent volume. Unlike hard surface rendering, Volumetric rendering evaluates light rays as they pass through the volume. This generally means evaluating **Opacity** and a **Color** for each pixel that intersects the volume.

---
## Stratified Sampling
Check [wiki] for detail information. Rather than simply drawing samples at regular spacing, the stratified sampling approach allows the model to sample a continuous space, therefore conditioning the network to learn over a continuous space. More things about ray sampling could be found [here](https://docs.nerf.studio/en/latest/nerfology/model_components/visualize_samplers.html).

---
## Spatial Distortions
When rendering a target view of a scene, the camera will emit a camera ray for each pixel and query the scene at points along this ray. We can choose where to query these points using different samplers. These samplers have some notion of bounds that define where the ray should start and terminate. If you know that everything in your scenes exists within some predefined bounds (ie. a cube that a room fits in) then the sampler will properly sample the entire space. If however the scene is unbounded (ie. an outdoor scene) defining where to stop sampling is challenging. One option to increase the far sampling distance to a large value (ie. 1km). Alternatively **we can warp the space into a fixed volume**. Below are supported distortions.

---
## Position Encoding
A more detail tutorial could be found [here](https://dtransposed.github.io/blog/2022/08/06/NeRF/) for using position encoding to capture details of Radiance Fields. 

![Encoding](./images/position%20encoding.png)

---
## Multilayer Perceptron
The Multilayer Perceptron was developed to tackle this limitation. It is a neural network where the mapping between inputs and output is non-linear.

A Multilayer Perceptron has input and output layers, and **one or more hidden layers** with many neurons stacked together. And while in the Perceptron the neuron must have an activation function that imposes a threshold, like ReLU or sigmoid, neurons in a Multilayer Perceptron can use any arbitrary activation function.

![MLP](./images/mlp.png)

---
# Peak Signal-to-noise Ratio
Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Because many signals have a very wide dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel scale.

PSNR is commonly used to quantify reconstruction quality for images and video subject to lossy compression.

![PSNR](./images/psnr.png)


