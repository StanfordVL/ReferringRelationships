# ************************************* CONFIG PARAMETERS *************************************
params = {
    "session_params": {
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 15,
        "save_dir": "/data/chami/ReferringRelationships/09_20_2017",
        "models_dir": "/data/chami/ReferringRelationships/models"
    },
    "model_params": {
        "embedding_dim": 100,
        "hidden_dim": 100,
        "feat_map_dim": 14,
        "input_dim": 224,
        "num_subjects": 100,
        "num_predicates": 70,
        "num_objects": 100
    },
    "data_params": {
        "train_data_dir": "/data/chami/VRD/09_09_2017/val/",
        "val_data_dir": "/data/chami/VRD/09_09_2017/val/",
        "image_data_dir": "/data/chami/VRD/sg_dataset/sg_train_images/"
    }#,
#    "eval_params": {
#        "score_thresh": [0.5, 0.7, 0.9]
#    }
}
