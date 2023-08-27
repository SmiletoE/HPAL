import glob
import json
import os
from os.path import join

import numpy as np


class S3DIS:
    def __init__(self, test_area_idx, log_file, cfg):
        self.log_file = log_file
        self.name = 'S3DIS'
        self.path = cfg.data_path
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = []

        for area in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']:
            cur_dir = os.path.join(self.path, area, 'coords')
            files = glob.glob(join(cur_dir, '*.npy'))
            self.all_files += files

        print(self.all_files)

        f_l = open(cfg.init_labeled_data, 'r')
        self.labeled_points = json.load(f_l)
        f_l.close()

        self.input_xyz = {'train': [], 'validation': []}
        self.input_colors = {'train': [], 'validation': []}
        self.input_labels = {'train': [], 'validation': []}
        self.input_names = {'train': [], 'validation': []}

        self.load_sub_sampled_clouds()

    def load_sub_sampled_clouds(self):
        for i, file_path in enumerate(self.all_files):
            xyz_path = file_path
            colors_path = file_path.replace('coords', 'rgb')
            label_path = file_path.replace('coords', 'labels')

            area = file_path.split('/')[-3]
            cloud_name = file_path.split('/')[-1][:-4]
            sp_key_name = area + '#' + cloud_name

            if sp_key_name in self.labeled_points:
                cloud_split = 'train'
                print('train:' + sp_key_name)
            else:
                cloud_split = 'validation'
                print('val:' + sp_key_name)

            self.input_xyz[cloud_split] += [xyz_path]
            self.input_colors[cloud_split] += [colors_path]
            self.input_labels[cloud_split] += [np.load(label_path)]
            self.input_names[cloud_split] += [sp_key_name]
