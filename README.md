# :cyclone: Literature of Novel View Synthesis
This respository is mainted as collections of state-of-art academic papers, code, reports, and datasets for novel view synthesis with implicit representations. 

- [:milky\_way: Neural Radiance Field (NeRF)](#milky_way-neural-radiance-field-nerf)
- [:black\_joker: Multiple Sphere Images (MSIs)](#black_joker-multiple-sphere-images-msis)
- [:jigsaw: Signed Distance Function (SDF) / Depth-Based](#jigsaw-signed-distance-function-sdf--depth-based)
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
- [Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://zju3dv.github.io/enerf/), Lin et al., Siggraph Asia 2022 | [github] (https://github.com/zju3dv/ENeRF)
  
</details>

<details open>
<summary>Pose</summary>

- [NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior](https://nope-nerf.active.vision/), Bian et al., CVPR 2023 
- [F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories](https://totoro97.github.io/projects/f2-nerf/), Wang et al., CVPR 2023
</details>

<details open>
<summary>Reconstruction & Lighting</summary>


- [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/), Wang et al., NeurIPS 2021 | [github](https://github.com/Totoro97/NeuS)
- [Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures], Metzer et al., | [github](https://github.com/eladrich/latent-nerf)
- [Light Field Neural Rendering](https://light-field-neural-rendering.github.io/), Suhail et al., CVPR 2022 | [github](https://github.com/google-research/google-research/tree/master/light_field_neural_rendering)

</details>

---
# :black_joker: Multiple Sphere Images (MSIs)
Papers in this section are not so convenient to group because of the less popularity compared to the NeRF. Research in MSIs is quite useful and meaningful for panorama video applications, especially 6 DOF VR videos. 

The most profound papers originates from multiplane images (MPIs), which are "[Single-view view synthesis with multiplane images](https://single-view-mpi.github.io/)" from CVPR 2020 and "[Stereo Magnification: Learning view synthesis using multiplane images](https://tinghuiz.github.io/projects/mpi/)" from Siggraph 2018.

<details open>
<summary>Quality & Fast Inference</summary>

- [MatryODShka: Real-time 6DoF Video
View Synthesis using Multi-Sphere Images](https://visual.cs.brown.edu/projects/matryodshka-webpage/), Attal et al., ECCV 2020
- [SOMSI: Spherical Novel View Synthesis
with Soft Occlusion Multi-Sphere Images](https://tedyhabtegebrial.github.io/somsi/), Habtegebrial et al., CVPR 2022 | [github](https://github.com/tedyhabtegebrial/SoftOcclusionMSI)
- [Immersive Light Field Video
With A Layered Mesh Representation](https://augmentedperception.github.io/deepviewvideo/), Broxton et al., Siggraph 2020
- MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects, Peng et al., ECCV 2022 | [github](https://github.com/JuewenPeng/MPIB)
</details>

# :jigsaw: Signed Distance Function (SDF) / Depth-Based
SDF as a kind of implicit representations for 3D scenes is a quite popular method recently for reconstruction tasks for novel-view enerations because it could be optimized with differentiable rendering frameworks.

<details open>
<summary>Rendering Quality & Speed</summary>

- [Multiview Neural Surface Reconstruction
by Disentangling Geometry and Appearance](https://lioryariv.github.io/idr/), Yariv et al., NeurIPS 2020 | [github](https://github.com/lioryariv/idr)

- [Extracting Triangular 3D Models, Materials, and Lighting From Images](https://nvlabs.github.io/nvdiffrec/), Munkberg et al., CVPR 2022 | [github](https://github.com/NVlabs/nvdiffrec)

</details>

<details open>
<summary>6 DOF </summary>

- [360∘ Stereo Image Composition with Depth Adaption](https://arxiv.org/abs/2212.10062), Huang et al., 
- [Casual 6-DoF: free-viewpoint panorama using a handheld 360 camera](https://arxiv.org/abs/2203.16756) Chen et al., 

</details>

---
# :dart: Dataset
<details open>
<summary>Normal 2D</summary>

- [Shiny - MultiView](https://drive.google.com/drive/folders/1kYGyIJI6AduHC-bM312N41WPjAoYf8Um)
- [ENerf - MultiView](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md)
- [Nerf - MultiView](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- [Deep Voxel - 3D Scans](https://github.com/vsitzmann/deepvoxels)
- [RealEstate10K - Camera Poses](https://google.github.io/realestate10k/)

<details open>
<summary>360 Panorama</summary>

- [SegFuse - RGBD](https://github.com/HAL-lucination/segfuse)

</details>

<details open>
<summary>Synthetic Data</summary>

- [Objaverse - Objects](https://huggingface.co/datasets/allenai/objaverse)
- [Replica - Indoor Scenes](https://github.com/facebookresearch/Replica-Dataset)
- [Hypersim - Indoor Scenes](https://github.com/apple/ml-hypersim)


</details>

---
# :art: Implementations
<details open>
<summary>NeRF</summary>

- [Implement NeRF with Pytorch-Lightning](https://github.com/kwea123/nerf_pl/)
- [Implement Instant-ngp Nerf with Taichi](https://github.com/taichi-dev/taichi-nerfs)

</details>


<details open>
<summary>Deep Learning FrameWorks</summary>

- [labml.ai Deep Learning Paper Implementations](https://nn.labml.ai/index.html) | [github](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- [Learn To Reproduce Papers: Beginner’s Guide](https://towardsdatascience.com/learn-to-reproduce-papers-beginners-guide-2b4bff8fcca0)

<details open>
<summary>3D Libraries</summary>

- [Taichi](https://github.com/taichi-dev/taichi)
- [Google / Visu3d](https://github.com/google-research/visu3d)
- [Nvidia / Warp](https://github.com/NVIDIA/warp)

</details>
