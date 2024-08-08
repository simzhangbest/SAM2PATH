from box import Box

config = {
    "batch_size": 8,
    "accumulate_grad_batches": 2,
    "num_workers": 8,
    "out_dir": "/root/workspace/code/sam-path/SAMPath/",
    "opt": {
        "num_epochs": 60,
        "learning_rate": 5e-4,
        "weight_decay": 1e-2, #1e-2,
        "precision": 32, # "16-mixed"
        # "precision": "16-mixed",
        "steps":  [72 * 25, 72 * 29],
        "warmup_steps": 72,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/root/workspace/code/sam-path/pretrained/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,

        "extra_encoder": 'uni_v1',
        "extra_type": "fusion",
        "extra_checkpoint": "/root/workspace/code/sam-path/pretrained/uni/pytorch_model.bin",
    },
    "loss": {
        "focal_cof": 0.25,
        "dice_cof": 0.75,
        "ce_cof": 0.0,
        "iou_cof": 0.0625,
    },
    "dataset": {
        "dataset_root": "/root/workspace/code/sam-path/path_data/BCSS/merged_dataset",
        "dataset_csv_path": "/root/workspace/code/sam-path/SAMPath/dataset_cfg/BCSS_cv.csv",
        "data_ext": ".png",
        "val_fold_id": 0,
        "num_classes": 6,

        "ignored_classes": (0),
        "ignored_classes_metric": None, # if we do not count background, set to 1 (bg class)
        "image_hw": (1024, 1024), # default is 1024, 1024

        "feature_input": False, # or "True" for *.pt features
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std": (0.229, 0.224, 0.225),
    }
}

cfg = Box(config)