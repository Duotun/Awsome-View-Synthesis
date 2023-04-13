<!-- omit in toc -->
# :cyclone: Instant NGP with Taichi
This document records the basic understanding and implementations for Instant NGP with [Taichi](https://github.com/taichi-dev/taichi-nerfs).

- Instant NGP: 
  1) "Occupency Grid" for accelerations of NeRF's training
  2) "Hash Encoder" for acceleraitons of NeRF's convergence rate 
![Hash Table](images/hash_table.png)


- NeRF Pipline:
![NeRF Pipeline](./images/nerf_struct.png)
![AutoGrad and AutoDiff](./images/autograd.png)

- Ray Marching for Rendering Results:
   - Early Stop with High Opacity
![Ray Marching](./images/ray_marching.png)
![Early Stop](./images/early_stop.png)
  