#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.bin.asr_inference import Speech2Text
from src.bin.asr_inference_maskctc import Speech2Text as Speech2TextMaskCTC

import sys
import torch
from src.utils import *

import os
import yaml
import argparse
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from colorama import Fore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to study the influence of each branch in each layer of the Branchformer Encoder.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model-config-file", required=True, type=str, help="Path to a config file that specifies the VSR model architecture.")
    parser.add_argument("--val-accs", required=True, type=str, help="The CSV file where, per each epoch, we have the WER and CER for the validation set")
    parser.add_argument("--average-epochs", default=30, type=int, help="Number of epochs which will be considered to compute the average model.")
    parser.add_argument("--output-dir", required=True, type=str, help="Path where the average model will be store.")

    args = parser.parse_args()

    # -- configuration architecture details
    model_config_file = Path(args.model_config_file)
    with model_config_file.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    model_config = argparse.Namespace(**model_config)

    # -- loading end-to-end speech recogniser
    if model_config.model == "espnet":
        speech2text = Speech2Text(
            asr_train_config= args.model_config_file,
            asr_model_file=None,
            lm_train_config=None,
            lm_file=None,
            **model_config.inference_conf,
        )
    elif model_config.model == "maskctc":
        speech2text = Speech2TextMaskCTC(
            asr_train_config=args.model_config_file,
            asr_model_file=None,
            token_type=model_config.token_type,
            bpemodel=model_config.bpemodel,
            **model_config.inference_conf,
        )

    val_accs = pd.read_csv(args.val_accs, delimiter=",").sort_values(by=["wer", "cer"])
    checkpoint_paths = val_accs[:args.average_epochs]["model_check_path"].tolist()

    print(checkpoint_paths)

    average_model(speech2text.asr_model, checkpoint_paths)
    save_model(args.output_dir, speech2text.asr_model, "average_"+str(args.average_epochs).zfill(3))
