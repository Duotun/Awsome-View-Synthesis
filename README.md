
<!-- omit in toc -->
# :cyclone: Literature of Awesome View Synthesis
This repository is mainted as collections of state-of-art academic papers, code, reports, and datasets for novel view synthesis with implicit representations. 


- [:milky\_way: Neural Radiance Field (NeRF)](#milky_way-neural-radiance-field-nerf)
- [:black\_joker: Multiple Sphere Images (MSIs)](#black_joker-multiple-sphere-images-msis)
- [:jigsaw: Signed Distance Function (SDF) / Depth-Based](#jigsaw-signed-distance-function-sdf--depth-based)
- [:dart: Dataset](#dart-dataset)
- [:art: Implementations](#art-implementations)

---
# :milky_way: Neural Radiance Field (NeRF)

An overall and detailed collection of NeRF could be found in this ["Awesome Neural Radiance Fields"](https://github.com/awesome-NeRF/awesome-NeRF) repository. 

Here is another collection of CVPR 2023 NeRF related papers: [NeRF-CVPR 2023](https://github.com/lif314/NeRFs-CVPR2023).

<details open>
<summary>Origins of NeRF</summary>

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf), Mildenhall et al., ECCV 2020 |  [![Star](https://img.shields.io/github/stars/bmild/nerf.svg?style=social&label=Star)](https://github.com/bmild/nerf) 
- NeRF++: Analyzing and Improving Neural Radiance Fields, Zhang et al., |  [![Star](https://img.shields.io/github/stars/Kai-46/nerfplusplus.svg?style=social&label=Star)](https://github.com/Kai-46/nerfplusplus)
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/), Barron et al., ICCV 2021 |  [![Star](https://img.shields.io/github/stars/google/mipnerf.svg?style=social&label=Star)](https://github.com/google/mipnerf)
- [NeRF in the Wild](https://nerf-w.github.io/), Brualla et al., CVPR 2021 

</details>

<details open>
<summary>Dynamic Scenes</summary>

- [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://www.albertpumarola.com/research/D-NeRF/index.html), Pumarola et al., CVPR 2021 |  [![Star](https://img.shields.io/github/stars/albertpumarola/D-NeRF.svg?style=social&label=Star)](https://github.com/albertpumarola/D-NeRF)
- PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification, Li et al., ICLR 2023 |   [![Star](https://img.shields.io/github/stars/xuan-li/PAC-NeRF.svg?style=social&label=Star)](https://github.com/xuan-li/PAC-NeRF)
- [InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds](https://tijiang13.github.io/InstantAvatar/), Jiang et al.,
- [Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://zju3dv.github.io/enerf/), Lin et al., Siggraph Asia 2022 |  [![Star](https://img.shields.io/github/stars/zju3dv/ENeRF.svg?style=social&label=Star)](https://github.com/zju3dv/ENeRF)
- [NeRF-Supervised Deep Stereo](https://nerfstereo.github.io/), Tosi et al., CVPR 2023 | [![Star](https://img.shields.io/github/stars/zju3dv/ENeRF.svg?style=social&label=Star)](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)  
- [Instant-NVR: Instant Neural Volumetric Rendering for Human-object Interactions
from Monocular RGBD Stream](https://nowheretrix.github.io/Instant-NVR/) Jiang et al., CVPR 2023
</details>

<details open>
<summary>Pose</summary>

- [NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior](https://nope-nerf.active.vision/), Bian et al., CVPR 2023 
- [F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories](https://totoro97.github.io/projects/f2-nerf/), Wang et al., CVPR 2023
</details>

<details open>
<summary>Reconstruction & Lighting</summary>


- [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/), Wang et al., NeurIPS 2021 | [![Star](https://img.shields.io/github/stars/zju3dv/ENeRF.svg?style=social&label=Star)](https://github.com/Totoro97/NeuS)
- [Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures], Metzer et al., | [![Star](https://img.shields.io/github/stars/eladrich/latent-nerf.svg?style=social&label=Star)](https://github.com/eladrich/latent-nerf)
- [Light Field Neural Rendering](https://light-field-neural-rendering.github.io/), Suhail et al., CVPR 2022 | [![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star)](https://github.com/google-research/google-research/tree/master/light_field_neural_rendering)
- [TexIR: Multi-view Inverse Rendering for Large-scale Real-world Indoor Scenes](http://yodlee.top/TexIR/), Li et al., CVPR 2023 | [![Star](https://img.shields.io/github/stars/LZleejean/TexIR_code.svg?style=social&label=Star)](https://github.com/LZleejean/TexIR_code)
- [Neural Fields meet Explicit Geometric Representations for Inverse Rendering of Urban Scenes](https://nv-tlabs.github.io/fegr/), Wang et al., CVPR 2023 :star:
- [Neural Rendering in a Room: Amodal 3D Understanding and Free-Viewpoint Rendering for the Closed Scene Composed of Pre-Captured Objects](https://zju3dv.github.io/nr_in_a_room/), Yang et al., SIGGRAPH 2022 :star:

</details>

<details open>
<summary> Edit NeRF </summary>

- [Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field](https://zju3dv.github.io/sine/), Bao et al., CVPR 2023 :star: 



</details>

---
# :black_joker: Multiple Sphere Images (MSIs)
Papers in this section are not so convenient to group because of the less popularity compared to the NeRF. Research in MSIs is quite useful and meaningful for panorama video applications, especially 6 DOF VR videos. 

The most profound papers originates from multiplane images (MPIs), which are "[Single-view view synthesis with multiplane images](https://single-view-mpi.github.io/)" from CVPR 2020 and "[Stereo Magnification: Learning view synthesis using multiplane images](https://tinghuiz.github.io/projects/mpi/)" from Siggraph 2018.

<details open>
<summary>Quality & Fast Inference</summary>

- [MatryODShka: Real-time 6DoF Video View Synthesis using Multi-Sphere Images](https://visual.cs.brown.edu/projects/matryodshka-webpage/), Attal et al., ECCV 2020
- [SOMSI: Spherical Novel View Synthesis with Soft Occlusion Multi-Sphere Images](https://tedyhabtegebrial.github.io/somsi/), Habtegebrial et al., CVPR 2022 | [![Star](https://img.shields.io/github/stars/tedyhabtegebrial/SoftOcclusionMSI.svg?style=social&label=Star)](https://github.com/tedyhabtegebrial/SoftOcclusionMSI)
- [Immersive Light Field Video With A Layered Mesh Representation](https://augmentedperception.github.io/deepviewvideo/), Broxton et al., Siggraph 2020
- MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects, Peng et al., ECCV 2022 |  [![Star](https://img.shields.io/github/stars/JuewenPeng/MPIB.svg?style=social&label=Star)](https://github.com/JuewenPeng/MPIB)
- [3D Video Loops from Asynchronous Input](https://limacv.github.io/VideoLoop3D_web/), Li et al., CVPR 2023 |  [![Star](https://img.shields.io/github/stars/limacv/VideoLoop3D.svg?style=social&label=Star)](https://github.com/limacv/VideoLoop3D)

</details>

# :jigsaw: Signed Distance Function (SDF) / Depth-Based
SDF as a kind of implicit representations for 3D scenes is a quite popular method recently for reconstruction tasks for novel-view enerations because it could be optimized with differentiable rendering frameworks. However, we need to pay attention that parametrization of the SDF as a single fully-connected Multi-Layer Perceptron (MLP) often leads to overly smooth geometry and color.

Multiview RGB images coupled with depth images are always the efficient ways to do the mutltiview geometry tasks and we should never overlook this topic.

<details open>
<summary>Rendering Quality & Speed</summary>

- [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance](https://lioryariv.github.io/idr/), Yariv et al., NeurIPS 2020 |  [![Star](https://img.shields.io/github/stars/lioryariv/idr.svg?style=social&label=Star)](https://github.com/lioryariv/idr)
- [Extracting Triangular 3D Models, Materials, and Lighting From Images](https://nvlabs.github.io/nvdiffrec/), Munkberg et al., CVPR 2022 | [![Star](https://img.shields.io/github/stars/NVlabs/nvdiffrec.svg?style=social&label=Star)](https://github.com/NVlabs/nvdiffrec)
- [MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction](https://niujinshuchong.github.io/monosdf/), Yu et al., NeurIPs 2022 |  [![Star](https://img.shields.io/github/stars/autonomousvision/monosdf.svg?style=social&label=Star)](https://github.com/autonomousvision/monosdf)
- [PermutoSDF: Fast Multi-View Reconstruction with Implicit Surfaces using Permutohedral Lattices](https://radualexandru.github.io/permuto_sdf/) Rosu et al., CVPR 2023 |  [![Star](https://img.shields.io/github/stars/RaduAlexandru/permuto_sdf.svg?style=social&label=Star)](https://github.com/RaduAlexandru/permuto_sdf)
- [BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis](https://arxiv.org/abs/2302.14859) Yariv et al., CVPR 2023

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
- [Replica - Indoor Scenes](https://github.com/facebookresearch/Replica-Dataset) :star:
- [Hypersim - Indoor Scenes](https://github.com/apple/ml-hypersim)
- [Circle - Indoor Dynamic Scenes](https://github.com/Stanford-TML/circle_dataset)
- [InteriorVerse](https://interiorverse.github.io/) :star:

</details>

<details open>
<summary>Capture App</summary>

- [PolyCapture](https://poly.cam/)
- [KIRI Engine](https://www.kiriengine.com/)
- [RealityCapture from Epic](https://www.capturingreality.com/) :star:
</details>

---
# :art: Implementations
<details open>
<summary>NeRF</summary>

- [NeRF Studio](https://github.com/nerfstudio-project/nerfstudio) :star:
- [Implement NeRF with Pytorch-Lightning](https://github.com/kwea123/nerf_pl/) :star:
- [Implement Instant-ngp Nerf with Taichi](https://github.com/taichi-dev/taichi-nerfs)
- [Torch NGP](https://github.com/ashawkey/torch-ngp)
- [SDF Studio](https://github.com/autonomousvision/sdfstudio)

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
