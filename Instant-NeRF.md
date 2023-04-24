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
Check [wiki] for detail information. Rather than simply drawing samples at regular spacing, the stratified sampling approach allows the model to sample a continuous space, therefore conditioning the network to learn over a continuous space.


 