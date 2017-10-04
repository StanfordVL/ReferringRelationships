# ************************************* CONFIG PARAMETERS *************************************
params = {
    "session_params": {
        "opt": "rms",
        "lr": 0.0001,
        "lr_decay": 0.,
        "batch_size": 128,
        "epochs": 100,
        "save_dir": "test", #"/data/chami/ReferringRelationships/09_20_2017",
        "models_dir": "/data/chami/ReferringRelationships/models/09_24_2017", 
        "save_best_only": True
    },
    "model_params": {
        "embedding_dim": 128,
        "hidden_dim": 512,
        "feat_map_dim": 14,
        "input_dim": 224,
        "num_subjects": 100,
        "num_predicates": 70,
        "num_objects": 100,
        "p_drop": 0.2
    },
    "data_params": { 
        "train_data_dir": "/data/chami/VRD/09_20_2017/train/",
        "val_data_dir": "/data/chami/VRD/09_20_2017/val/",
        "image_data_dir": "/data/chami/VRD/sg_dataset/sg_train_images/"
    }
}
