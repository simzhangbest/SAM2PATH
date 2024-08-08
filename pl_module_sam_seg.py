from functools import partial

import torch
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import time

from losses import SAMLoss

class SamSeg(LightningModule):

    def __init__(
            self,
            cfg,
            sam_model: nn.Module,
            metrics: MetricCollection,
            num_classes: int,
            focal_cof: float = 20.,
            dice_cof: float = 1.,
            iou_cof: float = 1.,
            ce_cof: float = 0.,
            lr: float = 0.0001,
            weight_decay: float = 0.01,
            lr_steps: list = (10, 20),
            warmup_steps: int = 0,
            ignored_index=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sam_model", "metrics"]) # 这将自动记录所有通过 __init__ 传入的参数
        self.model = sam_model
        self.num_classes = num_classes

        self.loss = SAMLoss(focal_cof, dice_cof, ce_cof, iou_cof)

        self.train_metrics = metrics.clone(postfix='/train')
        self.valid_metrics = nn.ModuleList([metrics.clone(postfix='/val'), metrics.clone(postfix='/test')])
        self.test_metrics = metrics.clone(prefix='final_test/')

        self.lr = lr

        self.ignored_index = ignored_index

        self.time_and_cnt = [0., 0]

    def forward(self, images):
        # use forward for inference/predictions
        pred_masks, iou_predictions = self.model(images)

        # pred_masks and iou_predictions are lists  将list 变成 torch 张量
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        return pred_masks, iou_predictions

    def calc_loss(self, pred_masks, gt_masks, iou_predictions, ignored_masks):
        loss_dict = self.loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks)
        assert "loss" in loss_dict
        return loss_dict

    @torch.no_grad()
    def process_masks(self, gt_masks):
        # gt_cls_masks = [gt_masks == i for i in range(0, self.num_classes + 1)]

        ignored_masks = gt_masks == 0
        # gt_cls_masks = torch.stack(gt_cls_masks[1:], dim=1).float()
        ignored_masks = ignored_masks.unsqueeze(1).long()
        return gt_masks, ignored_masks

    def predict_mask(self, pred_masks, gt_masks, ignored_masks):
        # pred_masks = [batch_size, #classes, h, w]
        # note class 0 is always for ignored classes
        pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
        pred_masks = pred_masks * (1 - ignored_masks.squeeze(1))

        if self.ignored_index is not None:
            pred_masks[pred_masks == self.ignored_index] = 0
            gt_masks[gt_masks == self.ignored_index] = 0

        return pred_masks, gt_masks

    def training_step(self, batch, batch_idx):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks)

        self.log_losses(losses, "train")

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)
        self.train_metrics.update(mask_cls_pred, gt_masks)
        # self.train_metrics(mask_cls_pred, gt_masks)

        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        return losses["loss"]

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        prefix = get_prefix_from_val_id(dataloader_idx)
        metrics_idx = dataloader_idx if dataloader_idx is not None else 0

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks)

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)

        if not self.trainer.sanity_checking:
            self.log_losses(losses, prefix)
            self.valid_metrics[metrics_idx].update(mask_cls_pred, gt_masks)
            # self.valid_metrics[metrics_idx](mask_cls_pred, gt_masks)
            # self.log_dict(self.valid_metrics[metrics_idx], on_step=False, on_epoch=True,
            #               add_dataloader_idx=False)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            for valM in self.valid_metrics:
                self.log_dict(valM.compute(), add_dataloader_idx=False)
                valM.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)


        # pred_masks, iou_predictions = self(images)
        with torch.no_grad():
            time_start = time.perf_counter()
            pred_masks, iou_predictions = self.model(images)
            time_predict = time.perf_counter() - time_start

        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        self.time_and_cnt[0] += time_predict
        self.time_and_cnt[1] += 1
        print("Average prediction time: %f" % (self.time_and_cnt[0] / self.time_and_cnt[1]))

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)
        return mask_cls_pred

    def log_losses(self, losses, prefiex):
        if prefiex == "train":
            for t in losses:
                self.log("Loss/%s_%s" % (prefiex, t), losses[t], on_epoch=True, on_step=True, sync_dist=True)
        else:
            for t in losses:
                self.log("Loss/%s_%s" % (prefiex, t), losses[t], on_epoch=True, on_step=False, sync_dist=True,
                         add_dataloader_idx=False)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_steps, verbose=False)
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            elif step < self.hparams.lr_steps[0]:
                return 1.0
            elif step < self.hparams.lr_steps[1]:
                return 0.1
            else:
                return 0.01
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=False)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }#[optimizer], [scheduler]

def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    else:
        raise NotImplementedError

