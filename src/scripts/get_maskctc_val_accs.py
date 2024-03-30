#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.tasks.asr import ASRTask
from src.evaluation.Tasas import computeTasas
from src.bin.asr_inference import Speech2Text
from src.bin.asr_inference_maskctc import Speech2Text as Speech2TextMaskCTC

import os
import sys
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from colorama import Fore
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import *
from src.transforms.WavTransforms import *
from src.transforms.VisualTransforms import *

def inference(args, speech2text, eval_loader, dataset, epoch):
    print(f"Decoding {dataset.upper()} dataset using the checkpoint of the {epoch} epoch:")

    # -- obtaining hypothesis
    dst_dir = os.path.join(args.output_dir, "inferences/")
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, dataset+"_"+epoch+".inf")
    with open(dst_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for xs_pad, ilens, ys_pad, olens, refs in tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.YELLOW, Fore.RESET)):
                result = speech2text(torch.squeeze(xs_pad, 0))
                # -- dumping results
                hyp = result[0][0]
                f.write(refs[0].strip() + "#" + hyp.strip() + "\n")

    # -- computing WER
    wer, cer, ci_wer, ci_cer = computeTasas(dst_path)
    report_wer = "%WER: " + str(wer) + " ± " + str(ci_wer); print(f"\n{report_wer}")
    report_cer = "%CER: " + str(cer) + " ± " + str(ci_cer); print(report_cer)
    with open(dst_path.replace(".inf", ".wer"), "w", encoding="utf-8") as f:
        f.write(report_wer + "\n")
        f.write(report_cer + "\n")

    return wer, cer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Speech Recognition System based on an End-to-End architecture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--model-config-file", required=True, type=str, help="Path to a config file that specifies the model architecture")
    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")
    parser.add_argument("--modality", default="audio", type=str, help="Choose the modality: audio, video, and audiovisual.")
    parser.add_argument("--snr-target", default=9999, type=int, help="A specific signal-to-noise rate when adding noise to the audio waveform.")
    parser.add_argument("--checkpoints-dir", required=True, type=str, help="Path where the Mask-CTC-based ASR model checkpoints are")
    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the CSV files containing the validation accuracies")

    args = parser.parse_args()

    # -- preprocessing data
    if args.modality == "video":
        if "lip-rtve" in args.validation_dataset.lower():
            (mean, std) = (0.491, 0.166)
        elif "vlrf" in args.validation_dataset.lower():
            (mean, std) = (0.392, 0.142)
        else:
            (mean, std) = (0.421, 0.165)
        fps = 50 if "vlrf" in args.validation_dataset.lower() else 25

        eval_data_transforms = Compose([
            Normalise(0.0, 250.0),
            Normalise(mean, std),
            CenterCrop((88,88)),
        ])
    elif args.modality == "audio":
        eval_data_transforms = Compose([
            AddNoise(noise_path="./src/noise/babble_noise.wav", sample_rate=16000, snr_target=args.snr_target),
        ])

    # -- configuration architecture details
    model_config_file = Path(args.model_config_file)
    with model_config_file.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    override_yaml(model_config, args.yaml_overrides)
    model_config = argparse.Namespace(**model_config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(model_config.token_type, model_config.bpemodel, model_config.token_list)

    # -- building speech-to-text recoginiser
    val_inf_stats = []
    model_checkpoints = sorted(os.listdir(args.checkpoints_dir))
    for model_check in tqdm(model_checkpoints):
        epoch = model_check.split("_")[-1].split(".")[0]

        if int(epoch) > 2:
            model_check_path = os.path.join(args.checkpoints_dir, model_check)
            speech2text = Speech2TextMaskCTC(
                asr_train_config=args.model_config_file,
                asr_model_file=model_check_path,
                token_type=model_config.token_type,
                bpemodel=model_config.bpemodel,
                **model_config.inference_conf,
            )

            # -- -- creating development set
            val_loader = get_dataloader(model_config, dataset_path=args.validation_dataset, transforms=eval_data_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality=args.modality)

            # -- -- infering development set
            epoch = model_check.split("_")[-1].split(".")[0]
            wer, cer = inference(args, speech2text, val_loader, "val", epoch)
            val_inf_stats.append( (model_check_path, wer, cer) )

        # -- creating csv file
        df = pd.DataFrame(val_inf_stats, columns=["model_check_path", "wer", "cer"])
        df.to_csv(os.path.join(args.output_dir, "../val_inf_stats.csv"))
