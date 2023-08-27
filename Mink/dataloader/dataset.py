import copy
import numpy as np
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from torchsparse import SparseTensor

import Mink.dataloader.transforms as t


class Stanford3DDataset:
    ROTATION_AXIS = 'z'

    def __init__(self, xyz, colors, labels, file_names, voxel_size, labeled_points=None):
        self.xyz = xyz
        self.colors = colors
        self.labels = labels
        self.file_names = file_names
        self.voxel_size = voxel_size
        self.labeled_points = labeled_points

        self.use_augs = {'scale': True, 'rotate': True, 'elastic': True, 'chromatic': True}

        self.prevoxel_aug_func = self.build_prevoxel_aug_func()
        self.postvoxel_aug_func = self.build_postvoxel_aug_func()

    def __getitem__(self, idx):
        # Read data
        coords = np.load(self.xyz[idx]).astype(np.float32)
        # feats = np.load(self.colors[idx]) * 255.0
        feats = np.load(self.colors[idx]).astype(np.float32)
        # labels = np.load(self.labels[idx]).astype(np.int32)
        labels = self.labels[idx].astype(np.int32)

        if self.labeled_points is not None:
            labels_cp = np.ones_like(labels) * -100
            labels_cp.astype(np.int32)

            name = self.file_names[idx]
            labeled_points_idx = self.labeled_points[name]

            labels_cp[labeled_points_idx] = labels[labeled_points_idx]

            labels = labels_cp

        # =======================================no augmentation=================================================
        lidarOrigin, labelsOrigin, labels_Origin, inverse_mapOrigin = self.voxelize(coords, feats, labels, False, False)

        # =======================================strong augmentation==============================================
        lidarStrongAug, labelsStrongAug, labels_StrongAug, inverse_mapStrongAug = self.voxelize(coords, feats, labels,
                                                                                                True, True)

        return {
            'lidar_Origin': lidarOrigin,
            'targets_Origin': labelsOrigin,
            'targets_mapped_Origin': labels_Origin,
            'inverse_map_Origin': inverse_mapOrigin,
            'lidar_StrongAug': lidarStrongAug,
            'targets_StrongAug': labelsStrongAug,
            'targets_mapped_StrongAug': labels_StrongAug,
            'inverse_map_StrongAug': inverse_mapStrongAug,
            'file_name': self.file_names[idx]
        }

    def voxelize(self, coords, feats, labels, is_prevoxel_aug, is_postvoxel_aug):
        coords = copy.deepcopy(coords)
        feats = copy.deepcopy(feats)
        labels = copy.deepcopy(labels)

        # Prevoxel Augmentation
        if is_prevoxel_aug:
            coords, feats, labels = self.prevoxel_aug_func(coords, feats, labels)

        # Voxelize
        pc_ = np.round(coords / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        # Postvoxel transformation
        if is_postvoxel_aug:
            pc_, feats, labels = self.postvoxel_aug_func(pc_, feats, labels)

        labels = labels.reshape(-1)
        labels_ = labels

        feats /= 255.0
        feat_ = np.concatenate([feats, coords], axis=1)

        # Sparse Quantize
        inds, labels, inverse_map = sparse_quantize(pc_, feat_, labels_, return_index=True, return_invs=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return lidar, labels, labels_, inverse_map

    def collate_fn(self, inputs):
        return sparse_collate_fn(inputs)

    def __len__(self):
        return len(self.xyz)

    def build_prevoxel_aug_func(self):
        aug_funcs = []

        if self.use_augs.get('elastic', False):
            aug_funcs.append(
                t.RandomApply([
                    t.ElasticDistortion([(0.2, 0.4), (0.8, 1.6)])
                ], 0.95)
            )
        if self.use_augs.get('rotate', False):
            aug_funcs += [
                t.Random360Rotate(self.ROTATION_AXIS, around_center=True),
                t.RandomApply([
                    t.RandomRotateEachAxis([(-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (0, 0)])
                ], 0.95)
            ]
        if self.use_augs.get('scale', False):
            aug_funcs.append(
                t.RandomApply([t.RandomScale(0.9, 1.1)], 0.95)
            )
        if self.use_augs.get('translate', False):
            # Positive translation should do at the end. Otherwise, the coords might be in negative space
            aug_funcs.append(
                t.RandomApply([
                    t.RandomPositiveTranslate([0.2, 0.2, 0])
                ], 0.95)
            )
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None

    def build_postvoxel_aug_func(self):
        aug_funcs = []
        if self.use_augs.get('dropout', False):
            aug_funcs.append(
                t.RandomApply([t.RandomDropout(0.2)], 0.5),
            )
        if self.use_augs.get('hflip', False):
            aug_funcs.append(
                t.RandomApply([t.RandomHorizontalFlip(self.ROTATE_AXIS)], 0.95),
            )
        if self.use_augs.get('chromatic', False):
            # The feats input should be in [0-255]
            aug_funcs += [
                t.RandomApply([t.ChromaticAutoContrast()], 0.2),
                t.RandomApply([t.ChromaticTranslation(0.1)], 0.95),
                t.RandomApply([t.ChromaticJitter(0.05)], 0.95)
            ]
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None


