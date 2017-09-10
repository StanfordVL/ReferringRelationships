# ************************************* CONFIG PARAMETERS *************************************
params = {
    "session_params": {
        "lr": 0.005,
        "batch_size": 8,
        "epochs": 20,
        "save_dir": "/data/chami/ReferringRelationships/models/09_09_2017",
        "models_dir": "/data/chami/ReferringRelationships/models"
    },
    "model_params": {
        "embedding_dim": 100,
        "hidden_dim": 14,
        "feat_map_dim": 14,
        "input_dim": 224,
        "num_subjects": 100,
        "num_predicates": 70,
        "num_objects": 100
    },
    "data_params": {
        "train_data_dir": "/data/chami/VRD/overfit/train/",
        "val_data_dir": "/data/chami/VRD/overfit/val/",
        "image_data_dir": "/data/chami/VRD/sg_dataset/sg_train_images/"
    },
    "eval_params": {
        "score_thresh": 0.9,
        "iou_thresh": 0.5
    }
}
