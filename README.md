# :cyclone: Literature of Novel View Synthesis
Collections of State-of-Art Academic Papers, Code, Reports, and Datasets for Novel View Synthesis with Implicit Representations

- [:cyclone: Literature of Novel View Synthesis](#cyclone-literature-of-novel-view-synthesis)
- [:milky\_way: Neural Radiance Field (NeRF)](#milky_way-neural-radiance-field-nerf)
- [:black\_joker: Multiple Sphere Images (MSIs)](#black_joker-multiple-sphere-images-msis)
- [:jigsaw: Signed Distance Function (SDF)](#jigsaw-signed-distance-function-sdf)
- [:dart: Dataset](#dart-dataset)
- [:art: Implementations](#art-implementations)

---
# :milky_way: Neural Radiance Field (NeRF)

An overall and detailed collections of NeRF could be found in this ["Awesome Neural Radiance Fields"](https://github.com/awesome-NeRF/awesome-NeRF) respository.

<details open>
<summary>Origins of NeRF</summary>

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf), Mildenhall et al., ECCV 2020 | [github](https://github.com/bmild/nerf) 
- NeRF++: Analyzing and Improving Neural Radiance Fields, Zhang et al., | [github](https://github.com/Kai-46/nerfplusplus)
- [Mip-NeRF: A Multiscale Representation
for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/), Barron et al., ICCV 2021 | [github](https://github.com/google/mipnerf)

</details>

<details open>
<summary>Dynamic Scenes</summary>

- [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://www.albertpumarola.com/research/D-NeRF/index.html) | [github](https://github.com/albertpumarola/D-NeRF)
- PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification, Li et al., ICLR 2023 | [github](https://github.com/xuan-li/PAC-NeRF)
- [InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds](https://tijiang13.github.io/InstantAvatar/), Jiang et al.,

</details>

<details open>
<summary>Pose</summary>

- [NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior](https://nope-nerf.active.vision/), Bian et al., CVPR 2023 
- [F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories](https://totoro97.github.io/projects/f2-nerf/), Wang et al., CVPR 2023
</details>

<details open>
<summary>Reconstruction & Lighting</summary>


- [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/) | [github](https://github.com/Totoro97/NeuS)
- [Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures], Metzer et al., | [github](https://github.com/eladrich/latent-nerf)

</details>

---
# :black_joker: Multiple Sphere Images (MSIs)
Papers in this section are not so convenient to group because of the less popularity compared to the NeRF. Research in MSIs is quite useful and meaningful for panorama video applications, especially 6 DOF VR videos. 

should revist the papers on mpi

<details open>
<summary>Quality & Fast Inference</summary>

- [MatryODShka: Real-time 6DoF Video
View Synthesis using Multi-Sphere Images](https://visual.cs.brown.edu/projects/matryodshka-webpage/)

</details>

# :jigsaw: Signed Distance Function (SDF)
An overall deepsdf, bakedsdf

nvidia reconstruction nvdiff

---
# :dart: Dataset
<details open>
<summary>Normal 2D</summary>

</details>

<details open>
<summary>360 Panorama</summary>

</details>

<details open>
<summary>Synthetic Data</summary>


</details>

---
# :art: Implementations
<details open>
<summary>NeRF</summary>

- [Implement NeRF with Pytorch-Lightning](https://github.com/kwea123/nerf_pl/)
- [Implement Instant-ngp Nerf with Taichi](https://github.com/taichi-dev/taichi-nerfs)

</details>

<details open>
<summary>MSIs</summary>

</details>

<details open>
<summary>SDF</summary>
nvidia reconstruction
</details>

<details open>
<summary>3D Libraries</summary>

- [Taichi](https://github.com/taichi-dev/taichi)
- [Google / Visu3d](https://github.com/google-research/visu3d)
- [Nvidia / Warp](https://github.com/NVIDIA/warp)

</details>