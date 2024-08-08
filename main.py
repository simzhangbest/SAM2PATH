from argparse import ArgumentParser
import os
import time
# sim added
import timm
import torch
# from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection, JaccardIndex, F1Score, Dice
from network.sam_network_2 import PromptSAM, PromptSAMLateFusion, PromptSAMLateFusion_NOKAN
# from network.sam_network_backup import PromptSAM, PromptSAMLateFusion
from pl_module_sam_seg import SamSeg
import albumentations
# from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger

# project_path = "/mnt/zmy/code/sam-path/SAMPath"
# name = "SAM-PATH-CRAG"
# tb_writer_summary_path = os.path.join(project_path, "run", name, "Logs")
# current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
# log_dir = os.path.join(tb_writer_summary_path, current_time)
# logger = TensorBoardLogger(log_dir, name='my_model')

# import os
# os.environ["WANDB_MODE"]="offline"


def get_augmentation(cfg):
    # W, H = cfg.dataset.image_hw if cfg.dataset.image_hw is not None else (1024, 1024)
    W, H = cfg.dataset.image_hw
    transform_train_fn = albumentations.Compose([
        albumentations.RandomResizedCrop(H, W, scale=(0.08, 1.0), p=1.0),
        albumentations.Flip(p=0.75),
        albumentations.RandomRotate90(),
        albumentations.ColorJitter(0.1, 0.1, 0.1, 0.1),
    ])
    # transform_test_fn = None #albumentations.Compose([])
    transform_test_fn = albumentations.Compose([
        albumentations.Resize(H, W),
    ])
    # transform_train = lambda x: transform_train_fn(image=x[0], mask=x[1])["image"]
    # transform_test = lambda x: transform_test_fn(image=x)["image"]
    # return transform_train, transform_test
    return transform_train_fn, transform_test_fn


def get_metrics(cfg):
    num_classes = cfg.dataset.num_classes + 1 # Note that we have an extra class
    # if cfg.dataset.ignored_classes_metric is not None:
    #     ignore_index = [0, cfg.dataset.ignored_classes_metric]
    # else:
    ignore_index = 0
    metrics = MetricCollection({
        "IOU_Jaccard_Bal": JaccardIndex(num_classes=num_classes, ignore_index=ignore_index, task='multiclass'),
        "IOU_Jaccard": JaccardIndex(num_classes=num_classes, ignore_index=ignore_index, task='multiclass',
                                        average="micro"),
        "F1": F1Score(num_classes=num_classes, ignore_index=ignore_index, task='multiclass', average="micro"),
        "Dice": Dice(num_classes=num_classes, ignore_index=ignore_index, average="micro"),
        "Dice_Bal": Dice(num_classes=num_classes, ignore_index=ignore_index, average="macro"),
    })
    return metrics



def get_model(cfg):
    if cfg.model.extra_encoder is not None:  # cfg.model.extra_encoder  hipt
        print("Using %s as an extra encoder" % cfg.model.extra_encoder)
        neck = True if cfg.model.extra_type == 'plus' else False
        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)
        elif cfg.model.extra_encoder == 'uni_v1': # sim added
            from network.get_network import get_uni
            extra_encoder = get_uni(cfg.model.extra_checkpoint, neck=neck)
        else:
            raise NotImplementedError
    else:
        extra_encoder = None
    if cfg.model.extra_type in ['plus']:
        MODEL = PromptSAM
    elif cfg.model.extra_type in ['fusion']:
        MODEL = PromptSAMLateFusion
    else:
        raise NotImplementedError

    model = MODEL(
        model_type = cfg.model.type,
        checkpoint = cfg.model.checkpoint,
        prompt_dim = cfg.model.prompt_dim,
        num_classes = cfg.dataset.num_classes,
        extra_encoder = extra_encoder,
        freeze_image_encoder = cfg.model.freeze.image_encoder,
        freeze_prompt_encoder = cfg.model.freeze.prompt_encoder,
        freeze_mask_decoder = cfg.model.freeze.mask_decoder,
        mask_HW = cfg.dataset.image_hw,
        feature_input = cfg.dataset.feature_input,
        prompt_decoder = cfg.model.prompt_decoder,
        dense_prompt_decoder=cfg.model.dense_prompt_decoder,
        no_sam=cfg.model.no_sam if "no_sam" in cfg.model else None
    )
    return model

def get_data_module(cfg):
    from image_mask_dataset import GeneralDataModule, ImageMaskDataset, FtMaskDataset
    augs = get_augmentation(cfg)
    common_cfg_dic = {
        "dataset_root": cfg.dataset.dataset_root,
        "dataset_csv_path": cfg.dataset.dataset_csv_path,
        "val_fold_id": cfg.dataset.val_fold_id,
        "data_ext": ".jpg" if "data_ext" not in cfg.dataset else cfg.dataset.data_ext,
        "dataset_mean": cfg.dataset.dataset_mean,
        "dataset_std": cfg.dataset.dataset_std,
        "ignored_classes": cfg.dataset.ignored_classes,  # only supports None, 0 or [0, ...]
    }
    if cfg.dataset.feature_input is True:
        dataset_cls = FtMaskDataset
    else:
        dataset_cls = ImageMaskDataset

    data_module = GeneralDataModule(common_cfg_dic, dataset_cls, cus_transforms=augs,
                                    batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return data_module



def get_pl_module(cfg, model, metrics):
    pl_module = SamSeg(
        cfg = cfg,
        sam_model = model,
        metrics = metrics,
        num_classes = cfg.dataset.num_classes,
        focal_cof = cfg.loss.focal_cof,
        dice_cof = cfg.loss.dice_cof,
        ce_cof=cfg.loss.ce_cof,
        iou_cof = cfg.loss.iou_cof,
        lr = cfg.opt.learning_rate,
        weight_decay = cfg.opt.weight_decay,
        lr_steps =  cfg.opt.steps,
        warmup_steps=cfg.opt.warmup_steps,
        ignored_index=cfg.dataset.ignored_classes_metric,
    )
    return pl_module

def main(cfg):
    
    # sim added: logger module
    project_path = cfg.out_dir
    name = cfg.name
    tb_writer_summary_path = os.path.join(project_path, "run", name, "Logs")
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_dir = os.path.join(tb_writer_summary_path, current_time)
    logger = TensorBoardLogger(log_dir, name='v100_model')

    
    data_module = get_data_module(cfg)

    sam_model = get_model(cfg)

    metrics = get_metrics(cfg=cfg)

    pl_module = get_pl_module(cfg, model=sam_model, metrics=metrics)  # 包装给 pl

    # logger = WandbLogger(project=cfg.project, name=cfg.name, save_dir=cfg.out_dir, log_model=True)
    

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    accumulate_grad_batches = cfg.accumulate_grad_batches if "accumulate_grad_batches" in cfg else 1

    trainer = Trainer(default_root_dir=cfg.out_dir, logger=logger,
                      devices=cfg.devices,
                      max_epochs=cfg.opt.num_epochs,
                      accelerator="gpu", #strategy="auto",
                      #strategy='ddp_find_unused_parameters_true',
                      log_every_n_steps=20, num_sanity_val_steps=0,
                      precision=cfg.opt.precision,
                      callbacks=[lr_monitor],
                      accumulate_grad_batches=accumulate_grad_batches,
                      fast_dev_run=False)

    trainer.fit(pl_module, data_module)


if __name__ == '__main__':
    
# python main.py --config configs.CRAG --devices 1 --project sampath --name crag_run0
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(False)
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs.CRAG")
    parser.add_argument('--devices', type=lambda s: [int(item) for item in s.split(',')], default=[1])
    parser.add_argument('--project', type=str, default="sim-sampath-crag")
    parser.add_argument('--name', type=str, default="crag_run4_kan_sam")
    # parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    module = __import__(args.config, globals(), locals(), ['cfg'])
    cfg = module.cfg

    cfg["project"] = args.project
    cfg["devices"] = args.devices
    cfg["name"] = args.name
    # cfg["seed"] = args.seed

    # seed_everything(cfg["seed"])
    print(cfg)
    main(cfg)
    # print(cfg)
