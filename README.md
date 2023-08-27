Skeleton Based Action Recognition Methods
=============

#### - Skeleton-based action recognition
  - A computer vision task that involves recognizing human actions from a sequence of 3D skeletal joint data
  - The goal of skeleton-based action recognition is to develop algorithms that can understand and classify human actions from skeleton data
  - The gcn-based method (extracting visual features) and the cnn-based method (extracting graph features) are mainly used.

#### - Goals of this repository
  - Many skeleton-based action-recognition-methods implementation/test/customization etc.


Methods List
=============

#### - 2s-AGCN

```
@article{2s-AGCN,
  title={Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author={Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition, 2019},
  year={2019}
}
```

  - Implementation/Test/Customization codes link: https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/tree/2s-AGCN

#### - PoseC3D

```
@article{PoseC3D,
  title={Revisiting Skeleton-based Action Recognition},
  author={Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, Bo Dai},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition, 2022},
  year={2022}
}
```

  - Implementation/Test/Customization codes link: https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/tree/PoseC3D


Datasets
=============

#### - NTU RGB+D

  - ﻿﻿Download link: https://github.com/shahroudy/NTURGB-D
  
  - Data volume and classes
    - 56,000 action clips in 60 action classes
    - Kind of classes
      - NTU RGB+D: A1 to A60, NTU RGB+D 120: A1 to A120

<p align="center">
<img src="https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods/assets/52263269/b75d08bf-3061-4a2b-a3a7-ededa3acdcd2" width="85%"></img> 
</p>
  
  - Details of data
    - Each action is captured by 3 cameras at the same height but from different horizontal angles: −45, 0, 45
    - 3D joint locations of each frame detected by Kinect depth sensors
    - 25 joints for each subject in the skeleton sequences, while each video has no more than 2 subjects
    
  - Cross-subject (X-Sub) Train/Valid subset
    - Training set: 40,320 clips
    - validation set: 16,560 
    - The actors in the two subsets are different

  - Cross-view (X-View) Train/Valid subset
    - Training set (captured by cameras 2 and 3): 37,920 clips
    - validation set (captured by camera 1): 18,960

#### - Kinetics-skeleton

  - ﻿﻿Download link: https://github.com/open-mmlab/mmskeleton/blob/master/doc/START_RECOGNITION.md
  
  - Data volume and classes
    - 300,000 videos clips in 400 classes from YouTube videos
    - Kind of classes: refer to https://arxiv.org/pdf/1705.06950.pdf
    
  - Details of data
    - Raw video clips without skeleton data: estimate the locations of 18 joints on every frame of the clips using the publicly available OpenPose toolbox
  
  - Train/Valid subset
    - Training set: 240,000 clips
    - Validation set: 20,000 clips
    - Filtering some data using samples_with_missing_skeletons.txt

    
