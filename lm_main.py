#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from espnet2.tasks.lm import LMTask
from espnet2.torch_utils.model_summary import model_summary

import os
import sys
import yaml
import argparse
from tqdm import tqdm
from colorama import Fore
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import *

def training(lm, train_loader, optimizer, scheduler, accum_grad):
    lm.train()
    train_loss = 0.0

    # -- training
    optimizer.zero_grad()
    for batch_idx,(xs_pad, ilens) in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        xs_pad, ilens = xs_pad.to(lm_config.device), ilens.to(lm_config.device)

        loss = lm(xs_pad, ilens)[0] / accum_grad
        loss.backward()

        # -- updating
        if ((batch_idx+1) % accum_grad == 0) or (batch_idx+1 == len(train_loader)):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss / (len(train_loader) / accum_grad)

def validation(lm, data_loader):
    lm.eval()
    data_loss = 0.0

    # -- forward pass
    with torch.no_grad():
        for xs_pad, ilens in tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            xs_pad, ilens = xs_pad.to(lm_config.device), ilens.to(lm_config.device)

            loss, stats, weight = lm(xs_pad, ilens)
            data_loss += loss.item()

    return round(data_loss / len(data_loader), 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model based on an End-to-End architecture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--training-dataset", default="", type=str, help="Path to where the training dataset split is")
    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")

    parser.add_argument("--mode", default="both", type=str, help="Choose: 'training', 'evaluation' or 'both'")
    parser.add_argument("--lm-config-file", default="", type=str, help="Path to a config file that specifies the LM architecture")
    parser.add_argument("--load-lm", default="", type=str, help="Path to load a pre-trained LM")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the language model")

    args = parser.parse_args()

    # -- configuration architecture details
    lm_config_file = Path(args.lm_config_file)
    with lm_config_file.open("r", encoding="utf-8") as f:
        lm_config = yaml.safe_load(f)
    override_yaml(lm_config, args.yaml_overrides)
    lm_config = argparse.Namespace(**lm_config)

    # -- security checks
    security_checks(asr_config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(lm_config.token_type, lm_config.bpemodel, lm_config.token_list)

    # -- training
    if args.mode in ["training", "both"]:

        # -- -- building LM end-to-end system
        lm = LMTask.build_model(lm_config).to(
            dtype=getattr(torch, lm_config.dtype),
            device=lm_config.device,
        )
        print(model_summary(lm))

        # -- -- loading the LM end-to-end system from a checkpoint
        print(f"Loading the entire LM system from {checkpoint_path}")
        lm.load_state_dict(checkpoint)

        # -- -- creating dataloaders
        train_loader = get_lm_dataloader(lm_config, dataset_path=args.training_dataset, tokenizer=tokenizer, converter=converter, is_training=True)
        val_loader = get_lm_dataloader(lm_config, dataset_path=args.validation_dataset, tokenizer=tokenizer, converter=converter, is_training=False)
        test_loader = get_lm_dataloader(lm_config, dataset_path=args.test_dataset, tokenizer=tokenizer, converter=converter, is_training=False)

        # -- -- optimizer and scheduler
        optimizer, scheduler = set_optimizer(lm_config, lm, train_loader)

        # -- -- training process
        print("\nTRAINING PHASE\n")
        val_stats = []
        for epoch in range(1, lm_config.epochs+1):
            train_loss = training(lm, train_loader, optimizer, scheduler, lm_config.accum_grad)
            # -- look line 122 of https://github.com/espnet/espnet/blob/master/espnet2/lm/espnet_model.py
            # -- they are actually computing the perplexity without specifying a log base, so perfect!
            val_loss = validation(lm, val_loader)
            test_loss = validation(lm, test_loader)

            print(f"Epoch {epoch}: TRAIN LOSS={train_loss} || VAL LOSS={val_loss} || TEST LOSS={test_loss}")
            dst_check_path = save_model(args.output_dir, lm, str(epoch).zfill(3))
            val_stats.append( (dst_check_path, val_loss) )

        # -- -- computing average model
        save_val_stats(args.output_dir, val_stats)
        sorted_val_stats = sorted(val_stats, key=lambda x: x[1])
        check_paths = [check_path for check_path, ppl in sorted_val_stats[:lm_config.average_epochs]]
        average_model(lm, check_paths)
        save_model(args.output_dir, lm, "average")

    # -- evaluation
    if args.mode in ["evaluation"]:
        # -- -- building LM end-to-end system
        lm = LMTask.build_model(lm_config).to(
            dtype=getattr(torch, lm_config.dtype),
            device=lm_config.device,
        )
        print(model_summary(lm))

        # -- -- loading the LM end-to-end system from a checkpoint
        print(f"Loading the entire LM system from {checkpoint_path}")
        lm.load_state_dict(checkpoint)

        val_loader = get_lm_dataloader(lm_config, dataset_path=args.validation_dataset, tokenizer=tokenizer, converter=converter, is_training=False)
        test_loader = get_lm_dataloader(lm_config, dataset_path=args.test_dataset, tokenizer=tokenizer, converter=converter, is_training=False)

        val_loss = validation(lm, val_loader)
        test_loss = validation(lm, test_loader)

        print(f"VAL LOSS={val_loss} || TEST LOSS={test_loss}")


