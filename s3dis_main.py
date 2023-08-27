import argparse
import json
import os
from os.path import exists

import numpy as np
import random
import gc
import warnings

from active_func import generate_score, active_chose
from Mink.dataloader.dataset import Stanford3DDataset
from config import ConfigS3DIS as cfg
from helper_utils import log_out
from data_base import S3DIS

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
import torch
from Mink.base_agent import BaseTrainer as minkNet

warnings.filterwarnings('ignore')
np.random.seed(1)
random.seed(1)
torch.manual_seed(7122)
# torch.cuda.set_per_process_memory_fraction(0.50, 0)


def mk_dirs():
    os.makedirs(cfg.base_path) if not exists(cfg.base_path) else None
    os.mkdir(cfg.model_save_dir_student) if not exists(cfg.model_save_dir_student) else None
    os.mkdir(cfg.model_save_dir_teacher) if not exists(cfg.model_save_dir_teacher) else None
    os.mkdir(cfg.labeled_save_path) if not exists(cfg.labeled_save_path) else None
    os.mkdir(cfg.save_path_feat) if not exists(cfg.save_path_feat) else None
    os.mkdir(cfg.save_path_probs) if not exists(cfg.save_path_probs) else None


def train_fullsupervised():
    """
    Fully supervised baseline
    """
    model_mink = minkNet(cfg, Log_file, dataset)

    # Fully Supervised baseline with 100% labeled data(upperbound)
    train_dataset = Stanford3DDataset(dataset.input_xyz['train'], dataset.input_colors['train'],
                                      dataset.input_labels['train'], dataset.input_names['train'],
                                      voxel_size=0.05)

    # Fully Supervised baseline with sparsely labeled Data
    # train_dataset = Stanford3DDataset(dataset.input_xyz['train'], dataset.input_colors['train'],
    #                                         dataset.input_labels['train'], dataset.input_names['train'],
    #                                         labeled_points=dataset.labeled_points, voxel_size=0.05)

    val_dataset = Stanford3DDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                    dataset.input_labels['validation'], dataset.input_names['validation'],
                                    voxel_size=0.05)

    model_mink.train_SGD(train_dataset, val_dataset)


def train_HPAL():
    """
    our method: HPAL(Hierarchy Point-based Active Learning)
    """
    train_dataset = Stanford3DDataset(dataset.input_xyz['train'], dataset.input_colors['train'],
                                      dataset.input_labels['train'], dataset.input_names['train'],
                                      labeled_points=dataset.labeled_points, voxel_size=0.05)
    val_dataset = Stanford3DDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                    dataset.input_labels['validation'], dataset.input_names['validation'],
                                    voxel_size=0.05)

    # saving labeled data
    save_path_curlabeled = os.path.join(cfg.labeled_save_path, 'labeled_data_' + str(cfg.al_iter) + '.json')
    f1 = open(save_path_curlabeled, 'w')
    json.dump(dataset.labeled_points, f1)
    f1.close()

    if cfg.al_iter == 0:
        model_teacher.net.load_state_dict(model_student.net.state_dict())
    else:
        model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
        # model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)
        model_teacher.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)

    # Segmentation model training
    model_student.train_consistencyguide_semi_SGD(train_dataset, val_dataset, model_teacher)
    log_out('Iteration ' + str(cfg.al_iter) + ' training is completed', Log_file)

    model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
    model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)

    # Active learning
    score_final = generate_score(cfg, model_student, dataset, train_dataset, Log_file)
    log_out('scoring finish', Log_file)

    active_chose(cfg, score_final, dataset, log_file=Log_file)
    log_out('chosing finish', Log_file)

    del train_dataset
    del val_dataset
    del score_final
    gc.collect()


def test_any_model():
    model = minkNet(cfg, Log_file, dataset)
    val_dataset = Stanford3DDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                    dataset.input_labels['validation'], dataset.input_names['validation'],
                                    voxel_size=0.05)
    model.load_checkpoint(model_path, local_rank=0)
    model.test_s3dis(val_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='AL_train',
                        help='options: baseline_train, AL_train, test')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    Mode = FLAGS.mode
    test_area = FLAGS.test_area

    mk_dirs()
    Log_file = open(cfg.saving_path + '.txt', 'a')
    dataset = S3DIS(test_area, Log_file, cfg)

    if Mode == 'AL_train':
        model_student = minkNet(cfg, Log_file, dataset)
        model_teacher = minkNet(cfg, Log_file, dataset)

        while cfg.al_iter < cfg.max_iter:
            train_HPAL()
            cfg.al_iter += 1
    elif Mode == 'baseline_train':
        train_fullsupervised()
    elif Mode == 'test':
        model_path = FLAGS.model_path
        test_any_model()
    else:
        print('Please enter the right mode')
    print('finish')
