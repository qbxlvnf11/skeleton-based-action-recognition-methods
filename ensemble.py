import argparse
import pickle
import yaml
import os
import mmcv

import numpy as np
from tqdm import tqdm

from mmengine.config import Config
from pyskl.datasets import build_dataloader, build_dataset

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('config', help='config file path')
parser.add_argument('--work_dir_name', required=True, choices={'slowonly_r50_ntu60_xsub'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
args = parser.parse_args()

## Annotations
cfg = Config.fromfile(args.config)

dataset = build_dataset(cfg.data.test, dict(test_mode=True))

dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    shuffle=False)
dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
data_loader = build_dataloader(dataset, **dataloader_setting)

## Joints
work_dir_name = args.work_dir_name
joint_results_path = os.path.join('work_dirs', 'posec3d', work_dir_name, 'joint', 'joint_results.yaml')
with open(joint_results_path) as f:
    joint_results = yaml.load(f, Loader=yaml.Loader)
print(' ==> len(joint_results):', len(joint_results))
print(' ==> joint_results[0].shape:', joint_results[0].shape)

## Limbs
limb_results_path = os.path.join('work_dirs', 'posec3d', work_dir_name, 'limb', 'limb_results.yaml')
with open(limb_results_path) as f:
    limb_results = yaml.load(f, Loader=yaml.Loader)
print(' ==> len(limb_results):', len(limb_results))
print(' ==> limb_results[0].shape:', limb_results[0].shape)

right_num = total_num = right_num_5 = 0

for d in tqdm(zip(data_loader, joint_results, limb_results)):
    l = d[0]['label']
    joint_result = d[1]
    limb_result = d[2]
    r = joint_result + limb_result * args.alpha
    # print(r.argsort())
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    # print(r, l)
    right_num += int(r == int(l))
    total_num += 1

acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1_acc: {:.4f}%'.format(acc * 100))
print('top5_acc: {:.4f}%'.format(acc5 * 100))