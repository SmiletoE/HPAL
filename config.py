#################################################################
# configuration for training
#################################################################


class ConfigS3DIS:
    # Active learning related
    # For "chosen_rate_AL" and "chosen_points_per_pc", only enable one of them according to the method you use in the "active_chose" function
    chosen_rate_AL = 0.02  # The selection ratio for each iteration in active loop(unit: %)
    # chosen_points_per_pc = 4  # The number of points selected for a single point cloud in each active iteration
    al_iter = 0  # Start iteration
    max_iter = 5  # Maximum number of iterations
    active_strategy = 'HMMU'  # Scoring strategy for active learning, including random, entropy, MMU, lc, HMMU(ours)

    # Training related
    gpu = 0
    max_steps = 60000  # Number of training steps
    stat_freq = 40  # Frequency of logging
    save_freq = 1000  # Frequency of model saving
    input_channel = 6  # Input channel: xyzrgb
    num_classes = 13  # Number of calsses
    ignore_idx = -100  # Ignore label during training
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1  # Initial learning rate
    ema_keep_rate = 0.955  # Ema keep rate for teacher-student model
    pseudo_threshold = 0.75  # The confidence threshold for filtering the pseudo-labels
    optimizer = 'CosineAnnealingLR'  # Learning rate optimization, 'CosineAnnealingLR' or 'PolyLR' in our experiments
    save_ts_together = False

    # Path related
    data_path = '/userHOME/yb/data/HPAL/s3dis'  # Processed data path
    init_labeled_data = 'data_preparation/init/s3dis/random0.02percent.json'  # Path of initial labelled data
    base_path = '/userHOME/yb/model/HPAL/S3DIS-0.1percent-paperinit'  # Path to save the training results

    # Paths for various results
    saving_path = base_path + '/learner'  # Log saving path
    model_save_dir_student = base_path + '/mink_pth_s'  # Saving path of student model
    model_save_dir_teacher = base_path + '/mink_pth_t'  # Saving path of teacher model
    labeled_save_path = base_path + '/labeled_data'  # Saving path of the labelled data after each iteration
    save_path_feat = base_path + '/feat'  # Feature saving path
    save_path_probs = base_path + '/probs'  # Prediction saving path
