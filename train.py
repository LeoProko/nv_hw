import argparse
import collections
import warnings
import itertools

import numpy as np
import torch

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    metrics = []

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    gen_optimizer = config.init_obj(
        config["optimizer"],
        torch.optim,
        filter(lambda p: p.requires_grad, model.generator.parameters()),
    )
    desc_optimizer = config.init_obj(
        config["optimizer"],
        torch.optim,
        itertools.chain(
            filter(lambda p: p.requires_grad, model.msd.parameters()),
            filter(lambda p: p.requires_grad, model.mpd.parameters()),
        ),
    )
    gen_lr_scheduler = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, gen_optimizer
    )
    desc_lr_scheduler = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, desc_optimizer
    )

    trainer = Trainer(
        model,
        metrics,
        gen_optimizer,
        desc_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        gen_lr_scheduler=gen_lr_scheduler,
        desc_lr_scheduler=desc_lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    with torch.autograd.set_detect_anomaly(True):
        main(config)
