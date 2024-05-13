import argparse
import torchvision
import torch.nn.functional as F

from .diffusion import (
    PointDiTDiffusion,
)
from .diffusion_unet import PointUnetDiffusion
from .dit import PointDiT
from .Unet1D import Unet1D

from .utils import generate_cosine_schedule,generate_linear_schedule


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="cosine",
        ema_decay=0.9999,
        ema_update_rate=1,
        loss_type="l2",
    )

    return defaults


def get_diffusion_from_args(args):
    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )
    if "airfoil" in args.project_name:
        
        # model = PointDiT(
        #     latent_size = args.feature_size,
        #     input_channels = 1,
        #     hidden_size=256,
        #     condition_size1=11,
        #     condition_size2=26,
        #     )
        model = Unet1D(
            dim=8,
            channels=1,
            dim_mults=(1, 2, 4, 8),
            resnet_block_groups=1
        )
        diffusion = PointUnetDiffusion(
            model, latent_size=args.feature_size, channels=1,
            betas=betas,
            ema_decay=args.ema_decay,
            ema_update_rate=args.ema_update_rate,
            ema_start=5000,
            loss_type=args.loss_type,
        )

    return diffusion 