2s-AGCN
=============

#### - ﻿Test & Customizing Pytorch 2s-AGCN code in [official repository](https://github.com/lshiwjx/2s-AGCN)

#### - ﻿2s-AGCN (Two-Stream Adaptive Graph Convolutional Network)
  
  - The topology of the graph in 2s-AGCN: uniformly or individually learned by the BP algorithm in an end-to-end manner
    - Increasing the flexibility of the model for graph construction
    - Bringing more generality to adapt to various data samples
    
  - Two-stream framework
    - Modeling both the first-order and the second-order information simultaneously
    
  - More details: https://blog.naver.com/qbxlvnf11/223176317587

#### - ﻿Adaptive graph convolutional layer﻿

<p align="center">
<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/901f569a-817c-435e-bfed-4db2bc375b2a" width="30%"></img> 
</p>

  - ﻿﻿Adjacency matrix $A_k$, $B_k$, $C_k$
    - $A_k﻿$: original normalized N × N adjacency matrix
    - $B_k﻿$: elements of this matrix are parameterized and optimized together with the other parameters in the training process
    - $C_k﻿$: data-dependent graph which learn a unique graph for each sample
   
<p align="center">
<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/f3fb1072-3003-494e-84b7-46ae57e3eb28" width="45%"></img> 
</p>

#### - Two-Stream Networks

  - ﻿﻿J-stream (Joint stream) & B-stream (Bone stream)
  - The scores of two streams are added to obtain the final prediction

<p align="center">
<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/447ae18e-533c-449d-a234-90b31187020f" width="65%"></img> 
</p>


Structures of Project Folders 
=============

#### - After download data, pre-trained weights of 2s-AGCN and data preparation

```
${ROOT}
            |   |-- main.py
            |   |-- ensemble.py
            |   |-- ...
            |   |-- data
            |   |   |   |-- nturgbd_raw
            |   |   |   |   |   |-- nturgb+d_skeletons
            |   |   |   |   |   |   |   |-- 56,880 files   
            |   |   |   |   |   |-- samples_with_missing_skeletons.txt
            |   |   |   |-- ntu
            |   |   |   |   |   |-- xsub
            |   |   |   |   |   |   |   |-- train_data_bone.npy
            |   |   |   |   |   |   |   |-- train_data_joint.npy
            |   |   |   |   |   |   |   |-- train_label.pkl
            |   |   |   |   |   |   |   |-- val_data_bone.npy
            |   |   |   |   |   |   |   |-- val_data_joint.npy
            |   |   |   |   |   |   |   |-- val_label.pkl
            |   |   |   |   |   |-- xview
            |   |   |   |   |   |   |   |-- train_data_bone.npy
            |   |   |   |   |   |   |   |-- train_data_joint.npy
            |   |   |   |   |   |   |   |-- train_label.pkl
            |   |   |   |   |   |   |   |-- val_data_bone.npy
            |   |   |   |   |   |   |   |-- val_data_joint.npy
            |   |   |   |   |   |   |   |-- val_label.pkl    
            |   |   |   |-- kinetics-skeleton  
            |   |   |   |   |   |-- kinetics_train 
            |   |   |   |   |   |   |   |-- 246,534 files          
            |   |   |   |   |   |-- kinetics_val 
            |   |   |   |   |   |   |   |-- 19,906 files  
            |   |   |   |   |   |-- kinetics_train_label.json 
            |   |   |   |   |   |-- kinetics_val_label.json 
            |   |   |   |-- kinetics       
            |   |   |   |   |   |-- train_data_bone.npy
            |   |   |   |   |   |-- train_data_joint.npy
            |   |   |   |   |   |-- train_label.pkl
            |   |   |   |   |   |-- val_data_bone.npy
            |   |   |   |   |   |-- val_data_joint.npy
            |   |   |   |   |   |-- val_label.pkl
            |   |   |   |-- pre_train
            |   |   |   |   |   |-- ki_agcn_bone
            |   |   |   |   |   |   |   |-- ki_agcn_bone-64-122070.pt
            |   |   |   |   |   |-- ki_agcn_joint
            |   |   |   |   |   |   |   |-- ki_agcn_joint-64-122070.pt
            |   |   |   |   |   |-- ntu_cs_agcn_bone
            |   |   |   |   |   |   |   |-- ntu_cs_agcn_bone-49-62600.pt
            |   |   |   |   |   |-- ntu_cs_agcn_joint
            |   |   |   |   |   |   |   |-- ntu_cs_agcn_joint-49-62600.pt
            |   |   |   |   |   |-- ntu_cv_agcn_bone
            |   |   |   |   |   |   |   |-- ntu_cv_agcn_bone-49-58800.pt
            |   |   |   |   |   |-- ntu_cv_agcn_joint
            |   |   |   |   |   |   |   |-- ntu_cv_agcn_joint-49-58800.pt
```


Download Pretrained Weights, Test Scores
=============

#### - Download 2s-AGCN (J-stream & B-stream model) weights of each dataset and score file
  - Password: 1234

http://naver.me/FGFs4Fw8


Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/2sAGCN_env
```

#### - Run docker environment

```
nvidia-docker run -it --gpus all --name 2sAGCN_env --shm-size=64G -p 8866:8866 -e GRANT_SUDO=yes --user root \
    -v {root_path}:/workspace/2s-AGCN \
    -w /workspace/2s-AGCN qbxlvnf11docker/2sAGCN_env bash
```
  

How to Data Preparation
=============

#### 1. Download dataset and set in the dataset according to the Structures of Project Folders

  - Details and download: https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/tree/main

#### 2. Build NTU-RGB+D (xsub & xview) joints action recognition dataset

  - Create ntu folder and save processed joint data file (.npy) & label file (.pkl)
  - Shape = (num_data, num_channel, max_frame, num_joint, num_person_out)
  
```
python ./data_gen/ntu_gendata.py --data_path ./data/nturgbd_raw/nturgb+d_skeletons --out_folder ./data/ntu \
    --ignored_sample_path ./data/nturgbd_raw/samples_with_missing_skeletons.txt
```

#### 3. Build Kinetics-Skeleton joint action recognition dataset

  - Create kinetics folder and save processed joint data file (.npy) & label file (.pkl)
  - Shape = (num_data, num_channel, max_frame, num_joint, num_person_out)
  
```
python ./data_gen/kinetics_gendata.py --data_path ./data/kinetics-skeleton --out_folder ./data/kinetics
```

#### 4. Build NTU-RGB+D (xsub & xview), Kinetics-Skeleton bone action recognition dataset

  - Save processed bone data file (.npy) of all dataset
  
```
python ./data_gen/gen_bone_data.py
```


How to Train 2s-AGCN
=============

#### - Train J-stream

  - NTU-RGB+D xsub
 
```
python main.py --config ./config/nturgbd-cross-subject/train_joint_custom.yaml
```

  - NTU-RGB+D xview
 
```
python main.py --config ./config/nturgbd-cross-view/train_joint_custom.yaml
```

  - Kinetics Skeleton
 
```
python main.py --config ./config/kinetics-skeleton/train_joint_custom.yaml
```

#### - Train B-stream

  - NTU-RGB+D xsub
 
```
python main.py --config ./config/nturgbd-cross-subject/train_bone_custom.yaml
```

  - NTU-RGB+D xview
 
```
python main.py --config ./config/nturgbd-cross-view/train_bone_custom.yaml
```

  - Kinetics Skeleton
 
```
python main.py --config ./config/kinetics-skeleton/train_bone_custom.yaml
```


Test Results (NTU-RGB+D xsub)
=============

#### - Test J-stream

  - Save prediction score file (epoch1_test_score.pkl) for testing two-stream networks

```
python main.py --config ./config/nturgbd-cross-subject/test_joint_custom.yaml --weights pre_train/ntu_cs_agcn_joint/ntu_cs_agcn_joint-49-62600.pt 
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/584a3c78-d7df-4ed7-ac55-2069b6db23ad" width="40%"></img> 

#### - Test B-stream

  - Save prediction score file (epoch1_test_score.pkl) for testing two-stream networks

```
python main.py --config ./config/nturgbd-cross-subject/test_bone_custom.yaml --weights pre_train/ntu_cs_agcn_bone/ntu_cs_agcn_bone-49-62600.pt
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/3407092e-c295-44cb-80ad-a5988679eccb" width="40%"></img> 

#### - Test Two-Stream Networks

  - Fused score of J-stream and B-stream

```
python ensemble.py --datasets ntu/xsub
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/80603f25-6094-40c9-beac-9986b3d53960" width="13%"></img>


Test Results (NTU-RGB+D xview)
=============

#### - Test J-stream

  - Save prediction score file (epoch1_test_score.pkl) for testing two-stream networks

```
python main.py --config ./config/nturgbd-cross-view/test_joint_custom.yaml --weights pre_train/ntu_cv_agcn_joint/ntu_cv_agcn_joint-49-58800.pt 
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/bc756376-36ed-47ca-a081-7d17b352ba92" width="40%"></img> 

#### - Test B-stream

  - Save prediction score file (epoch1_test_score.pkl) for testing two-stream networks

```
python main.py --config ./config/nturgbd-cross-view/test_bone_custom.yaml --weights pre_train/ntu_cv_agcn_bone/ntu_cv_agcn_bone-49-58800.pt
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/054aea11-6c86-43f6-a167-3c49f447bb4e" width="40%"></img> 

#### - Test Two-Stream Networks

  - Fused score of J-stream and B-stream

```
python ensemble.py --datasets ntu/xview
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/69291587-99b4-42b2-94bf-b3dea55eaf2c" width="13%"></img>  


Test Results (Kinetics Skeleton)
=============

#### - Test J-stream

  - Save prediction score file (epoch1_test_score.pkl) for testing two-stream networks

```
python main.py --config ./config/kinetics-skeleton/test_joint_custom.yaml --weights pre_train/ki_agcn_joint/ki_agcn_joint-64-122070.pt
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/110f911a-7cb0-4a43-b10d-bce57ef0b1af" width="40%"></img> 

#### - Test B-stream

  - Save prediction score file (epoch1_test_score.pkl) for testing two-stream networks

```
python main.py --config ./config/kinetics-skeleton/test_bone_custom.yaml --weights pre_train/ki_agcn_bone/ki_agcn_bone-64-122070.pt 
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/21a8b25d-1fa4-4619-a02f-40a12cf055f8" width="40%"></img> 

#### - Test Two-Stream Networks

  - Fused score of J-stream and B-stream

```
python ensemble.py --datasets kinetics
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/479a7b34-5a38-4fbf-ad2b-429eb6a7e73c" width="13%"></img> 


