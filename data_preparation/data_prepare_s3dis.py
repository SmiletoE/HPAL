# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import numpy as np
import os
import pickle
from sklearn.neighbors import KDTree

from helper_tool import DataProcessing as DP

# Data path (Need to modify)
STANFORD_3D_IN_PATH = '/userHOME/yb/data/origin/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/'  # Path of the downloaded s3dis
STANFORD_3D_OUT_PATH = '/userHOME/yb/data/HPASSL/s3dis'  # Save path of the processed data

# Error line in Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt
# could not convert string to float: '185\x00187'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Stanford3DDatasetConverter:
    name_to_labels = {'ceiling': 0,
                      'floor': 1,
                      'wall': 2,
                      'beam': 3,
                      'column': 4,
                      'window': 5,
                      'door': 6,
                      'chair': 7,
                      'table': 8,
                      'bookcase': 9,
                      'sofa': 10,
                      'board': 11,
                      'clutter': 12,
                      'stairs': 12}     # Stairs --> clutter (follow KPConv)

    @classmethod
    def read_txt(cls, txtfile):
        # Read txt file and parse its content.
        with open(txtfile) as f:
            pointcloud = []
            for line in f:
                try:
                    data = [float(li) for li in line.split()]
                    if len(data) < 3:
                        # Prevent empty line in some file
                        raise Exception("Line with less than 3 digits")
                    pointcloud += [data]
                except Exception as e:
                    print(e, txtfile, flush=True)
                    continue

        # Load point cloud to named numpy array.
        pointcloud = np.array(pointcloud).astype(np.float32)
        assert pointcloud.shape[1] == 6
        xyz = pointcloud[:, :3].astype(np.float32)
        rgb = pointcloud[:, 3:].astype(np.uint8)
        return xyz, rgb

    @classmethod
    def convert_to_npy(cls, root_path, out_path):
        """Convert Stanford3DDataset to NPY format.
        Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
        """

        txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
        for idx, txtfile in enumerate(txtfiles):
            print(f"{idx+1} / {len(txtfiles)}", flush=True)
            file_sp = os.path.normpath(txtfile).split(os.path.sep)
            target_path = os.path.join(out_path, file_sp[-3])

            # output filename
            out_coords = os.path.join(target_path, "coords", file_sp[-2] + '.npy')
            out_rgb = os.path.join(target_path, "rgb", file_sp[-2] + '.npy')
            out_labels = os.path.join(target_path, "labels", file_sp[-2] + '.npy')
            out_proj = os.path.join(target_path, "proj", file_sp[-2] + '_proj.pkl')  # Used when downsampling point clouds are requied

            os.makedirs(os.path.join(target_path, "coords"), exist_ok=True)
            os.makedirs(os.path.join(target_path, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)
            os.makedirs(os.path.join(target_path, "proj"), exist_ok=True)  # Used when downsampling point clouds are requied
            if os.path.exists(out_coords):
                print(out_coords, ' exists')
                continue

            annotation, _ = os.path.split(txtfile)
            subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
            coords, feats, labels = [], [], []
            for inst, subcloud in enumerate(subclouds):
                # Read ply file and parse its rgb values.
                xyz, rgb = cls.read_txt(subcloud)
                _, annotation_subfile = os.path.split(subcloud)
                clsidx = cls.name_to_labels[annotation_subfile.split('_')[0]]

                coords.append(xyz)
                feats.append(rgb)
                labels.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)

            if len(coords) == 0:
                print(txtfile, ' has 0 files.')
            else:
                # Concat
                coords = np.concatenate(coords, 0).astype(np.float32)
                feats = np.concatenate(feats, 0).astype(np.uint8)
                labels = np.concatenate(labels, 0).astype(np.uint8)

                # Used when downsampling is not requied
                # np.save(out_coords, coords)
                # np.save(out_rgb, feats)
                # np.save(out_labels, labels)

                ##########################downsampling point clouds################################################
                coords_min = np.amin(coords, axis=0)
                coords -= coords_min

                sub_grid_size = 0.04  # For s3dis, we downsample to 0.04 by default to get faster computation while satisfying MinkowskiNet
                sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(coords, feats, labels, sub_grid_size)

                search_tree = KDTree(sub_xyz)

                proj_idx = np.squeeze(search_tree.query(coords, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                with open(out_proj, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

                np.save(out_coords, sub_xyz)
                np.save(out_rgb, sub_colors)
                np.save(out_labels, sub_labels)
                ##########################downsampling point clouds################################################


if __name__ == '__main__':
    Stanford3DDatasetConverter.convert_to_npy(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)