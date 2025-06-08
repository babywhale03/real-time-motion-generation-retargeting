# Real-time Motion Generation & Retargeting
YAICON (Yonsei AI Conference Contest)

| ![Real-time Text-to-Motion](https://github.com/user-attachments/assets/536fd7f9-3d73-455a-8540-187f1b584a1c) | ![Motion Retargeting](https://github.com/user-attachments/assets/f8add2ec-69dd-4613-b5dd-82e04523ab65) |
|:--:|:--:|
| **Text-to-Motion** | **Motion Retargeting** |


## 📣 Project Overview 
Recent advances in human motion generation have been driven by diffusion models, enabling the synthesis of natural and meaningful movements based on user input. In particular, generating and retargeting motions that reflect human intent has significant potential across a wide range of applications.

In this project, we aim to:
- Generate human skeleton-based motions from text input provided by users.
- Retarget these motions to be compatible with real-world bipedal robot platforms.

This allows for intuitive, high-level control of humanoid robots through natural language commands.

## 🪄 Environment Setup
### 1. CLoSD 
  - Create a Conda env and setup the requirements

```
conda create -n closd python=3.8
conda activate closd
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

  - Download [Isaac GYM](https://developer.nvidia.com/isaac-gym), and install it to your env

```
conda activate closd
cd <ISSAC_GYM_DIR>/python
pip install -e .
```

### 2. motion_retargeting
- Create a Conda env and setup the requirements
````
conda create --name retargeting python=3.8.12
conda activate retargeting 
````
- Install [PyTorch 1.10.0](https://pytorch.org/) inside the conda environment.

````
cd motion_retargeting
pip install -e .
````

- Download Lafan1 dataset (Be sure to read and follow their license agreements, and cite accordingly.)
````S
mkdir data_preprocess/Lafan1_and_dog/Lafan1
````
Go to the [Lafan1 website](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and download the lafan1.zip. Then unzip it and put all the .bvh files into `data_preprocess/Lafan1_and_dog/Lafan1`



## 😎 Team Members

#### [이재은 (Team Leader)](https://github.com/babywhale03) | [봉지환](https://github.com/jihwan-b) | [정혜진](https://github.com/hjchung1) | [조남기]()


## 📋 Acknowledgement

This repository builds upon the following projects, which provided the foundation for the implementation:

- [CLoSD](https://github.com/GuyTevet/CLoSD): Closing the Loop between Simulation and Diffusion for multi-task character control
- [PAN Motion Retargeting](https://github.com/hlcdyy/pan-motion-retargeting): Pose-aware Attention Network for Flexible Motion Retargeting by Body Part
