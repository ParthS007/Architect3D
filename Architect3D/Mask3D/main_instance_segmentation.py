import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
torch.serialization.add_safe_globals([ModelCheckpoint, DictConfig, ContainerMetadata])

import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything
import torch

torch.autograd.set_detect_anomaly(True)


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id
    #cfg.general.checkpoint = "/work/courses/3dv/20/OpenArchitect3D/Mask3D/saved/final/epoch=249-val_mean_ap_50=0.000.ckpt"
    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg["trainer"][
            "resume_from_checkpoint"
        ] = f"{cfg.general.save_dir}/last-epoch.ckpt" #"/work/courses/3dv/20/OpenArchitect3D/Mask3D/scannet200_val.ckpt" #f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.test(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    #<GROUP20 START>
    """
    checkpoint = torch.load('/work/courses/3dv/20/OpenArchitect3D/Mask3D/scannet200_val.ckpt')
    state_dict = checkpoint["state_dict"]
    for key in list(state_dict.keys()):
        #if key in ["model.backbone.final.kernel", "model.backbone.final.bias", "model.class_embed_head.weight", "model.class_embed_head.bias"]:
        #    del state_dict[key]
        if key == "model.backbone.final.kernel":
            state_dict[key] = torch.rand(96, 2754)
        elif key == "model.backbone.final.bias":
            state_dict[key] = torch.rand(1, 2754)
        elif key == "model.class_embed_head.weight":
            state_dict[key] = torch.rand(2754, 128)
        elif key == "model.class_embed_head.bias":
            state_dict[key] = torch.rand(2754)
        else:
            pass
    #state_dict["criterion.empty_weight"] = torch.zeros(2753)
    checkpoint["state_dict"] = state_dict
    torch.save(checkpoint, '/work/courses/3dv/20/OpenArchitect3D/Mask3D/scannet200_val.ckpt')
    """
    _old_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _old_load(*args, **kwargs)

    torch.load = _patched_load
    #<GROUP20 END>

    main()
