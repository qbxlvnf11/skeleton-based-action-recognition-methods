import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import mmcv

def get_all_files_in_directory(directory):
    file_paths = []
    directory_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def mapping_keypoints_taekwondo(taekwondo_keypoints):
    keypoints = np.zeros((17, 2))
    scores = np.zeros((17, 1))
    joints = ["코", "왼쪽 눈", "오른쪽 눈", "왼쪽 귀", "오른쪽 귀", "왼쪽 어깨", "오른쪽 어깨", "왼쪽 팔꿈치", "오른쪽 팔꿈치", \
        "왼쪽 손목", "오른쪽 손목", "왼쪽 엉덩이", "오른쪽 엉덩이", "왼쪽 무릎", "오른쪽 무릎", "왼쪽 발목", "오른쪽 발목"]

    for i, j in enumerate(joints):
        keypoints[i] = [taekwondo_keypoints[j]["x"], taekwondo_keypoints[j]["y"]]
        scores[i] = taekwondo_keypoints[j]["view"]

    return keypoints, scores

def build_annotations(dataset_name, dataset_folder_path):

    if dataset_name == 'Goyang Taekwondo Dataset':
        
        label_idx_mapping_dict = {'기본준비':0, '내려헤쳐막기':1, '돌려차고 앞굽이하고 아래막기':2, '돌려차고 앞굽이하고 얼굴바깥막고 지르기':3, \
                        '두발당성차고 앞굽이하고 안막고 두번지르기':4, '뒤꼬아서고 두주먹젖혀지르기':5, '뒤꼬아서고 등주먹앞치기':6, \
                        '뒷굽이하고 거들어바깥막기':7, '뒷굽이하고 거들어아래막기':8, '뒷굽이하고 바깥막기':9, '뒷굽이하고 손날거들어바깥막기':10, \
                        '뒷굽이하고 손날거들어아래막기':11, '뒷굽이하고 손날바깥막기':12, '뒷굽이하고 안막기':13, '뛰어앞차고 앞굽이하고 안막고 두번지르기':14, \
                        '모아서고 보주먹':15, '범서고 바탕손거들어안막고 등주먹앞치기':16, '범서고 바탕손안막기':17, '범서고 손날거들어바깥막기':18,\
                        '범서고 안막기':19, '앞굽이하고 가위막기':20, '앞굽이하고 거들어세워찌르기':21, '앞굽이하고 당겨지르기':22, '앞굽이하고 두번지르기':23, \
                        '앞굽이하고 등주먹앞치기':24, '앞굽이하고 등주먹앞치기하고 안막기':25, '앞굽이하고 바탕손안막고 지르기':26, '앞굽이하고 손날얼굴비틀어막기':27, \
                        '앞굽이하고 아래막고 안막기':28, '앞굽이하고 아래막고 지르기':29, '앞굽이하고 아래막기':30, '앞굽이하고 안막고 두번지르기':31, \
                        '앞굽이하고 안막기':32, '앞굽이하고 얼굴막기':33, '앞굽이하고 얼굴바깥막고 지르기':34, '앞굽이하고 얼굴지르기':35, \
                        '앞굽이하고 엇걸어아래막기':36, '앞굽이하고 외산틀막기':37, '앞굽이하고 제비품안치기':38, '앞굽이하고 지르기':39, \
                        '앞굽이하고 팔꿈치거들어돌려치기':40, '앞굽이하고 팔꿈치돌려치고 등주먹앞치기하고, 지르기':41, '앞굽이하고 팔꿈치표적치기':42, \
                        '앞굽이하고 헤쳐막기':43, '앞서고 등주먹바깥치기':44, '앞서고 손날안치기':45, '앞서고 아래막고 지르기':46, '앞서고 아래막기':47, \
                        '앞서고 안막고 지르기':48, '앞서고 안막기':49, '앞서고 얼굴막기':50, '앞서고 지르기':51, '앞차고 뒷굽이하고 바깥막기':52, \
                        '앞차고 범서고 바탕손안막기':53, '앞차고 앞굽이하고 등주먹앞치기':54, '앞차고 앞굽이하고 아래막고 안막기':55, \
                        '앞차고 앞굽이하고 지르기':56, '앞차고 앞서고 아래막고 지르기':57, '앞차고 앞서고 지르기':58, '옆서고 메주먹내려치기':59, \
                        '옆차고 뒷굽이하고 손날거들어바깥막기':60, '주춤서고 손날옆막기':61, '주춤서고 옆지르기':62, '주춤서고 팔꿈치표적치기':63}
        print(f' ===> Label mapping dict: {label_idx_mapping_dict}')

        folder_mapping_dict = {'TL1':'TS2', 'TL2':'TS1', 'TL3':'TS3', 'TL4':'TS4', 'TL5':'TS5', 'TL6':'TS6', 'TL7':'TS7', 'TL8':'TS8', 'VL1':'VS1'}
        splits = ['1.Training', '2.Validation']

        results_dict = defaultdict(list)
        # save_sample_video_flag = True
        # vid_num = 0

        for s in splits:

            label_folder_path = os.path.join(dataset_folder_path, '01.데이터', s, '라벨링데이터')
            img_folder_path = os.path.join(dataset_folder_path, '01.데이터', s, '원천데이터')

            label_file_paths = get_all_files_in_directory(label_folder_path)
            label_file_paths_dict = defaultdict(dict)
            
            for label_file_path in label_file_paths:
                label_file_path_split = label_file_path.split('/')
                class_name = label_file_path_split[-3]
                sequence = label_file_path_split[-2]

                if not sequence in label_file_paths_dict[class_name]:
                    label_file_paths_dict[class_name][sequence] = []
                label_file_paths_dict[class_name][sequence].append(label_file_path)

            for class_name in tqdm(label_idx_mapping_dict.keys()):
                for sequence_name in label_file_paths_dict[class_name].keys():

                    label_file_paths = label_file_paths_dict[class_name][sequence_name]
                    label_file_paths.sort(reverse=True)

                    for i, label_file_path in enumerate(label_file_paths):
                        # print(f' ===> {label_file_path}')\
                        label_file_path_split = label_file_path.split('-')

                        if label_file_path_split[-1][0] == 'S': # Start frame
                            M_path_list = []
                            M_keypoints_list = []
                            M_scores_list = []

                            S = label_file_path
                            # S_img_path = os.path.join(img_folder_path, folder_mapping_dict[S[-4]], \
                            #     S[-3], S[-2], S[-1][:-4]+'.jpg')
                            S_label_dict = mmcv.load(S)
                            S_keypoints, S_scores = mapping_keypoints_taekwondo(S_label_dict['labelingInfo'][0]['pose']['location'])

                        elif label_file_path_split[-1][0] == 'M': # Middle frame
                            M = label_file_path
                            # M_img_path = os.path.join(img_folder_path, folder_mapping_dict[M[-4]], \
                            #     M[-3], M[-2], M[-1][:-4]+'.jpg')
                            M_label_dict = mmcv.load(M)
                            M_keypoints, M_scores = mapping_keypoints_taekwondo(M_label_dict['labelingInfo'][0]['pose']['location'])
                            M_path_list.insert(0, M)
                            M_keypoints_list.insert(0, M_keypoints)
                            M_scores_list.append(M_scores)
                        
                        elif label_file_path_split[-1][0] == 'E': # End frame
                            E = label_file_path
                            # E_img_path = os.path.join(img_folder_path, folder_mapping_dict[E[-4]], \
                            #     E[-3], E[-2], E[-1][:-4]+'.jpg')

                            ## Keypoints
                            E_label_dict = mmcv.load(E)
                            E_keypoints, E_scores = mapping_keypoints_taekwondo(E_label_dict['labelingInfo'][0]['pose']['location'])
                            
                            keypoints = []
                            keypoints.extend([S_keypoints])
                            keypoints.extend(M_keypoints_list)
                            keypoints.extend([E_keypoints])

                            scores = []
                            scores.extend([S_scores])
                            scores.extend(M_scores_list)
                            scores.extend([E_scores])    

                            keypoints = np.expand_dims(np.stack(keypoints, axis=0), axis=0).astype(np.float16)
                            scores = np.expand_dims(np.stack(scores, axis=0), axis=0).astype(np.float16)
                            # print(f' ===> {class_idx}, {keypoints.shape}, {scores.shape}')
                            total_frames = keypoints.shape[1]

                            M_path_list = []
                            M_keypoints_list = []
                            M_scores_list = []

                            ## Class
                            S_split = S.split('/')
                            class_idx = label_idx_mapping_dict[S_split[-3]]

                            num_person_raw = 1
                            img_shape = (1080, 1920) #(S_label_dict['labelingInfo'][0]['images']['height'], S_label_dict['labelingInfo'][0]['images']['width'])
                            frame_dir = 'sample'
                            filename = './sample.avi'
                            file_path = './sample.avi'

                            d = {"label": class_idx, "num_person_raw": num_person_raw, "total_frames": total_frames, \
                                "keypoint": keypoints, "keypoint_score":scores, "img_shape":img_shape, \
                                "frame_dir":frame_dir, "filename":filename, "file_path":file_path}
                            results_dict[s].append(d)
                            
                            # if save_sample_video_flag and vid_num % 1000 == 0:
                                

                            # vid_num += 1

    return results_dict, splits

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset_name', required=True, choices=['Goyang_Taekwondo_Dataset'])
    parser.add_argument('--dataset_folder_path', required=True)
    parser.add_argument('--save_folder_path', required=True)
                            
    args = parser.parse_args()

    print(f' ===> Dataset: {args.dataset_name}')
    print(f' ===> Dataset folder path: {args.dataset_folder_path}')

    results_dict, splits = build_annotations(dataset_name=args.dataset_name, dataset_folder_path=args.dataset_folder_path)

    for s in splits:
        save_path = os.path.join(args.save_folder_path, s+'.pkl')
        mmcv.dump(results_dict[s], save_path)
        print(f' ===> Save path: {save_path}')