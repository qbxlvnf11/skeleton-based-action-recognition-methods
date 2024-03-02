PoseC3D
=============

#### - ﻿Test & Customizing Pytorch PoseC3D code in [official repository](https://github.com/kennymckormick/pyskl/tree/main)

#### - 3D-CNN-Based PoseC3D: overcoming limitations of existing GCN-based methods
 
  - Robustness
    
    - GCN-based methods: Since it is greatly affected by the distribution shift of coordinates, completely different results can be output even with small perturbation in coordinates.
    - PoseC3D: It showed the powerful generalization performance even when feeding human skeletons data collected in different scenarios or other methods.
      
  - Interoperability
    
    - GCN-based methods: In the case of existing action recognition, performance has been improved by effectively combining RGB, optical flow, skeletons, etc., but the graphical form of the skeleton representation is difficult to combine, so there is a limitation that this method cannot be used.
    - PoseC3D: It showed that integrating with other modalities is possible.
      
  - Scalability

    - GCN-based methods: In the case of GCN, since all human joints are treated as nodes, the complexity of GCN scales increases linearly with the number of persons. This causes restrictions on action recognition involving multiple persons, such as group activity recognition.
    - PoseC3D: It showed that can handle a number of people without increasing computational overhead.
    
#### - PoseC3D

<p align="center">
<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/44168111-788b-4240-8c2f-f26999bfaf35" width="80%"></img> 
</p>

  - ﻿﻿Process
    
    - Heatmaps of joints or limbs are stacked along the temporal dimension, and pre-processing is applied to create 3D heatmap volumes.
    - Finally classify 3D heatmap volumes using 3D-CNN

  - Two families of 3D-CNN: PoseConv3D, RGBPose-Conv3D
      
  - Reducing redundancy
    
    - Subject Centered Cropping: After finding the smallest bounding box of 2D poses created along all frames, all frames are cropped according to the found box and resized to fit the target size.
    - Uniform Sampling: Uniform sampling strategy for 3D-CNNs that randomly selects one frame from each segment after dividing the video into $N$ segments with the same length.
      
  - More details: https://blog.naver.com/qbxlvnf11/223182550920

#### - PoseConv3D for Pose Modality

  - ﻿﻿Two modifications for applying 3D-CNN to skeleton-based action recognition

    - Down-sampling operations, an early stage of 3D-CNN, are eliminated
    - Shallower (fewer layers) and thinner (fewer channels) networks
      
  - Three backbone: 'C3D', 'SlowOnly' (Best), 'X3D'
      
<p align="center">
<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/5b8fddb4-74db-409d-84fe-fb62f7713814" width="40%"></img> 
</p>

#### - RGBPose-Conv3D for Ensemble of Pose Modality and RGB Modality

  - ﻿﻿Two-stream 3D-CNN: Pose modality & RGB modality
    
  - ​The two pathways are asymmetrical because the properties of the two modalities are different
    - Compared to the RGB Pathway, the pose pathway has a smaller channel width and depth.
  
  - Early-stage feature fusion: bidirectional lateral connections between the two pathways

  - To avoid overfitting, individual cross-entropy loss is used for each pathway.


Download Processing Dataset
=============

#### - Download Processing Dataset

https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md


Download Pretrained Weights, Test Scores
=============

#### - Download PoseC3D pretrained weights and score file
  - Password: 1234

http://naver.me/GCabATkW

#### - Download PoseC3D pretrained weights of mmaction

https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md

https://download.openmmlab.com/mmaction/skeleton/posec3d/k400_posec3d-041f49c6.pth


Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/pose_c3d_env:korean
```

#### - Run docker environment

```
nvidia-docker run -it --gpus all --name pose_c3d_env --shm-size=64G -p 8899:8899 -e GRANT_SUDO=yes \
     --user root -v {root path}:/workspace/PoseC3D \
     -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
     -w /workspace/PoseC3D qbxlvnf11docker/pose_c3d_env:korean bash
```

How to Build Custom Dataset (HRNet Label)
=============

#### 1. Create video action annotations text file

  - Create two text file of train & validation

```
file_path_1 \t label_1
file_path_2 \t label_2
...
```

#### 2. Build custom dataset pkl file

  - Build two pkl file of train & validation

```
bash tools/dist_run.sh tools/data/custom_2d_skeleton.py {total number of gpus} --gpu_num {using gpu number} \
    --video-action-annotations {text file path containing video path and corresponding action label} \
    --out {save path of annotation pkl file} \
    --max_len {max length of frames}
```

#### 3. Confirm labels of extracted joints throught visualization

```
python vis_2d_skeletons_labels.py --annotations_path .{path of annotation pkl file} \
    --save_folder {save folder of visualization videos} \
    --fps {fps}
```

#### 4. Modify configuration file to suit custom dataset

  - Refer to './configs/posec3d/slowonly_r50_custom_k400p/s1_joint_custom.py'


```
model = dict(
    ...
    cls_head=dict(
        ...
        num_classes={number of classes of custom dataset},
        ...
        ),
    ...
    )

...

ann_file_train = {path of train annotation pkl file}
ann_file_val = {path of valid annotation pkl file}

...

data = dict(
    ...
    train=dict(
        ...
        dataset=dict(type=dataset_type, ann_file=ann_file_train, pipeline=train_pipeline)
        ),
    val=dict(type=dataset_type, ann_file=ann_file_val, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file_val, pipeline=test_pipeline)
    )

...

```


How to Build Custom Dataset (Parsing Label File)
=============

#### - Build custom dataset pkl file

  - choices of dataset_name parameter: 'Goyang Taekwondo Dataset'
  
```
python convert_pkl_annotations.py --dataset_name {dataset name} \
    --dataset_folder_path {dataset folder file path} \
    --save_folder_path {save path of annotation pkl file}
```


Train PoseConv3D
=============

#### - Train Joint

  - NTU-RGB+D xsub
 
```
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_ntu60_xsub/joint.py {total number of gpus} --validate
```

  - NTU-RGB+D 120 xsub
 
```
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_ntu120_xsub/joint.py {total number of gpus} --validate
```

#### - Train Limb

  - NTU-RGB+D xsub
 
```
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_ntu60_xsub/limb.py {total number of gpus} --validate
```


Fine-Tune PoseConv3D
=============

#### - Set 'load_from', 'find_unused_parameters' attributes in config file

#### - Loading pre-trained weights of Kinetics 400 dataset, fine-tuning model of UCF101 dataset

```
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_ucf101_k400p/s1_joint_custom.py {total number of gpus} --validate
```

#### - Loading pre-trained weights of Kinetics 400 dataset, fine-tuning model of FineGYM dataset

```
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_gym_k400p/s1_joint_custom.py {total number of gpus} --validate
```

#### - Loading pre-trained weights of Kinetics 400 dataset, fine-tuning model of Goyang Taekwondo dataset

```
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_goyang_taekwondo_k400p/s1_joint_custom.py {total number of gpus} --validate
```


Run Demo
=============

#### - Demo a video 

```
PYTHONPATH="demo/.." python demo/demo_skeleton.py {input video path} {video save path} \
    --config {config file path} \
    --checkpoint {checkpoint file path} \
    --label-map {class name mapping file}
```

#### - Demo videos

  - Read video action annotations text file and save accuracy and prediction results of all videos with excel file

```
PYTHONPATH="demo/.." python demo/demo_skeleton_videos.py \
    --config {config file path} \
    --video-folder-path {input videos folder path} \
    --save-folder-path {save videos folder path} \
    --checkpoint {checkpoint file path} \
    --label-map {class name mapping file} \
    --video-action-annotations {text file path containing video path and corresponding action label}
```

PoseConv3D Test Results: NTU-RGB+D xsub
=============

#### - Test Joint

  - Save prediction score file for testing ensemble of joint & limb 

```
bash tools/dist_test.sh ./configs/posec3d/slowonly_r50_ntu60_xsub/joint.py \
    ./work_dirs/posec3d/slowonly_r50_ntu60_xsub/joint/{weights path} {total number of gpus} \
    --out ./work_dirs/posec3d/slowonly_r50_ntu60_xsub/joint/joint_results.yaml \
    --eval top_k_accuracy mean_class_accuracy
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/a7c397c7-850a-45dd-95f4-455244aebd90" width="20%"></img> 

#### - Test Limb

  - Save prediction score file for testing ensemble of joint & limb 

```
bash tools/dist_test.sh ./configs/posec3d/slowonly_r50_ntu60_xsub/limb.py \
    ./work_dirs/posec3d/slowonly_r50_ntu60_xsub/limb/{weights path} {total number of gpus} \
    --out ./work_dirs/posec3d/slowonly_r50_ntu60_xsub/limb/joint_results.yaml \
    --eval top_k_accuracy mean_class_accuracy
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/a6ff30d9-958e-408c-9793-a4bad120ce34" width="20%"></img> 

#### - Test Ensemble of Joint & Limb

  - Fused score of joint & limb

```
python ensemble.py ./configs/posec3d/slowonly_r50_ntu60_xsub/joint.py \
    --work_dir_name slowonly_r50_ntu60_xsub
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/0a43fa34-c5bb-4d8b-a46e-84fa60d3b12e" width="20%"></img>


PoseConv3D Test Results: NTU-RGB+D 120 xsub
=============

#### - Test Joint

  - Save prediction score file for testing ensemble of joint & limb

```
bash tools/dist_test.sh ./configs/posec3d/slowonly_r50_ntu120_xsub/joint.py \
    ./work_dirs/posec3d/slowonly_r50_ntu120_xsub/joint/{weights path} {total number of gpus} \
    --out ./work_dirs/posec3d/slowonly_r50_ntu120_xsub/joint/joint_results.yaml \
    --eval top_k_accuracy mean_class_accuracy
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/f691de0c-aba2-4fb1-adbb-f1a1e31cf16f" width="20%"></img> 


Fine-Tuning PoseConv3D Test Results: UCF101
=============

#### - Test Joint

  - Save prediction score file for testing ensemble of joint & limb

```
bash tools/dist_test.sh ./configs/posec3d/slowonly_r50_ucf101_k400p/s1_joint_custom.py \
    ./work_dirs/posec3d/slowonly_r50_ucf101_k400p/s1_joint/{weights path} {total number of gpus} \
    --out ./work_dirs/posec3d/slowonly_r50_ucf101_k400p/s1_joint/joint_results.yaml \
    --eval top_k_accuracy mean_class_accuracy
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/2865e7c2-6db5-4326-86ca-842f9a5b3e47" width="20%"></img> 


Fine-Tuning PoseConv3D Test Results: FineGYM
=============

#### - Test Joint

  - Save prediction score file for testing ensemble of joint & limb

```
bash tools/dist_test.sh ./configs/posec3d/slowonly_r50_gym_k400p/s1_joint_custom.py \
    ./work_dirs/posec3d/slowonly_r50_gym_k400p/s1_joint/{weights path} {total number of gpus} \
    --out ./work_dirs/posec3d/slowonly_r50_gym_k400p/s1_joint/joint_results.yaml \
    --eval top_k_accuracy mean_class_accuracy
```

<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/0c9ac566-b715-497d-b795-ea2ae9a650fd" width="20%"></img> 


