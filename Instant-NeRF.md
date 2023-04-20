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

- Ray Marching for Rendering Results:
   - Early Stop with High Opacity
![Ray Marching](./images/ray_marching.png)
![Early Stop](./images/early_stop.png)

---
## Volumetric Rendering
Here attached an implementation and tutorials of Volumetric Rendering in [Shadertoy](https://disigns.wordpress.com/portfolio/shadertoy-glsl-demos/) for practice and reviews.

Volume rendering is a technique for visualizing sampled functions of a colored semi transparent volume. Unlike hard surface rendering, Volumetric rendering evaluates light rays as they pass through the volume. This generally means evaluating **Opacity** and a **Color** for each pixel that intersects the volume.


 