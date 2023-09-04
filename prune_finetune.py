import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm import dist

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate

from segm.tm_prune import tm_prune, tm_prune_block


@click.command(help="")
@click.option("--prune", default=False, is_flag=True)
@click.option("--freeze-matrix", default=False, is_flag=True)
@click.option("--pretrain-dir", type=str, help="logging directory")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option("--iter-num", default=0, type=int)
def main(
    prune,
    freeze_matrix,
    pretrain_dir,
    log_dir,
    dataset,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    amp,
    resume,
    iter_num,
):
    # get configuration
    cfg = config.load_config()

    ptu.set_gpu_mode(True)
    # distributed.init_process()
    # start distributed mode
    device = torch.device('cuda')
    dist.init_distributed_mode(cfg)

    # set up configuration
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate

    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    prune_cfg = cfg['prune']
    if prune:
        lr = 2e-5
        num_epochs = 1
        eval_freq = 1

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // cfg['world_size']
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            cooldown_epochs=30,
            warmup_epochs=0,
            warmup_lr=0.00001,
            decay_rate=0.1,
            min_lr=1e-7,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
        prune_kwargs=prune_cfg,
    )

    log_dir = Path(log_dir)
    pretrain_dir = Path(pretrain_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = pretrain_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        checkpoint_path = pretrain_dir / "pruned_{}_{}.pth".format(prune_cfg['prune_rate'], prune_cfg['keep_rate'])

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"

    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print("Resuming training from checkpoint: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "merged_tokens" in checkpoint:
            merged_tokens = checkpoint["merged_tokens"]
            print('merged_tokens: ', merged_tokens)
        else:
            merged_tokens = None
        if "merge_list" in checkpoint:
            merge_list = checkpoint["merge_list"]
            print('merge_list:', merge_list)
        else:
            merge_list = []
        if "recover_list" in checkpoint:
            recover_list = checkpoint["recover_list"]
            print('recover_list:', recover_list)
        else:
            recover_list = []
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        if prune:
            variant["algorithm_kwargs"]["start_epoch"] = 0
        else:
            variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1

        # model
        net_kwargs = variant["net_kwargs"]
        net_kwargs["n_cls"] = n_cls
        net_kwargs["merged_tokens"] = merged_tokens
        net_kwargs["merge_list"] = merge_list
        net_kwargs["recover_list"] = recover_list
        model = create_segmenter(net_kwargs)
        model.to(device)

        # if not prune:
        #     optimizer.load_state_dict(checkpoint["optimizer"])

        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        assert 0, 'pre-trained model is necessary'
        # sync_model(log_dir, model)

    model_without_ddp = model
    if cfg['distributed']:
        model = DDP(model, device_ids=[cfg['gpu']], find_unused_parameters=True)
        model_without_ddp = model.module

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)

    lr_scheduler = create_scheduler(opt_args, optimizer)
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # save config
    variant_str = yaml.dump(variant)
    print("Configuration:\n{}".format(variant_str))
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    # model_without_ddp = model
    # if hasattr(model, "module"):
    #     model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print("Train dataset length: {}".format(len(train_loader.dataset)))
    print("Val dataset length: {}".format(len(val_loader.dataset)))
    print("Encoder parameters: {}".format(num_params(model_without_ddp.encoder)))
    print("Decoder parameters: {}".format(num_params(model_without_ddp.decoder)))

    best = 0.0
    is_best = False
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        if prune:
            model_without_ddp.encoder.set_compute_attn()

        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            iter_num
        )

        if prune:
            prune_rate = prune_cfg['prune_rate']
            keep_rate = prune_cfg['keep_rate']
            merge_list = prune_cfg['merge_list']
            recover_list = prune_cfg['recover_list']
            # merged_tokens = tm_prune(model_without_ddp.encoder, prune_rate, keep_rate, merge_list)
            merged_tokens = tm_prune_block(model_without_ddp.encoder, keep_rate, merge_list, recover_list)
            model_without_ddp.encoder.reset_compute_attn()
            model.to(device)

        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            eval_logger, scores = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
            )
            print("Stats [{}]:".format(epoch), eval_logger, flush=True)
            print("")
            miou = scores['mean_iou']
            if miou > best:
                best = miou
                is_best = True
            else:
                is_best = False
        # save checkpoint
        if dist.is_main_process():
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                merged_tokens=merged_tokens,
                merge_list=merge_list,
                recover_list=recover_list,
                #optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                #lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            if prune:
                log_path = log_dir / 'pruned_{}_{}.pth'.format(prune_rate, keep_rate)
            else:
                if is_best:
                    log_path = log_dir / 'best.pth'
                else:
                    log_path = log_dir / 'checkpoint.pth'
            torch.save(snapshot, log_path)

        if prune:
            distributed.barrier()
            distributed.destroy_process()
            # sys.exit(1)
            return

        # log stats
        if dist.is_main_process():
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{"train_{}".format(k): v for k, v in train_stats.items()},
                **{"val_{}".format(k): v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            with open(log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    distributed.barrier()
    distributed.destroy_process()
    # sys.exit(1)
    return


if __name__ == "__main__":
    main()
