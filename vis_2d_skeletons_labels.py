import glob
import os
from pyskl.smp import *
from pyskl.utils.visualize import Vis3DPose, Vis2DPose
from mmcv import load, dump
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--annotations_path', required=True)
parser.add_argument('--save_folder', required=True)
parser.add_argument('--fps', type=int, default=12)
parser.add_argument('--layout', default='coco')
                        
args = parser.parse_args()

# Visualize 2D Skeletons with the original RGB Video
annotations = load(args.annotations_path)
for i, anno in enumerate(annotations):
    file_path = anno['file_path']
    print(' ========= idx:', i, '=========')
    print(' ==> origin file name:', file_path)
    print(' ==> total frames:',anno['total_frames']) 
    print(' ==> shape of keypoint:',anno['keypoint'].shape) 
    print(' ==> shape of keypoint_score:',anno['keypoint_score'].shape) 
    #video_url = f"http://download.openmmlab.com/mmaction/pyskl/demo/nturgbd/{frame_dir}.avi"
    #download_file(video_url, frame_dir + '.avi')
    vid = Vis2DPose(anno, thre=0.2, out_shape=(540, 960), layout=args.layout, fps=args.fps, video=file_path)
    #vid.ipython_display()
    save_video_path = os.path.join(args.save_folder, file_path.split('/')[-1][:-4]+'_vis.mp4')
    vid.write_videofile(save_video_path, fps=args.fps, codec="libx264")
