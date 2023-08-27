import json
import os
import numpy as np

np.random.seed(0)

root = '/userHOME/yb/data/HPASSL/s3dis/'
save_path = 'init/s3dis/seed0_random0.02labelpts.json'

initial_labeled_dic = {}
init_points = 0
all_points = 0

# for area in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']:
for area in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']:
    cur_dir = os.path.join(root, area, 'coords')
    file_names = os.listdir(cur_dir)
    for name in file_names:
        cur_path_xyz = os.path.join(cur_dir, name)
        xyz = np.load(cur_path_xyz)

        labeled_num_cur_pc = round(len(xyz) * 0.0002)
        # labeled_num_cur_pc = 4

        # random initialization
        init_pts = np.argsort(np.random.rand(xyz.shape[0]))[: labeled_num_cur_pc]

        out = [False] * len(xyz)
        for i in init_pts:
            out[i] = True
            init_points += 1

        all_points += len(xyz)

        pc_name = area + '#' + name[:-4]

        initial_labeled_dic[pc_name] = out

f1 = open(save_path, 'w')
json.dump(initial_labeled_dic, f1)
f1.close()
print(init_points)
print(all_points)
