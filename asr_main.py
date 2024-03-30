#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.tasks.asr import ASRTask
from src.evaluation.Tasas import computeTasas
from src.bin.asr_inference import Speech2Text
from src.bin.asr_inference_maskctc import Speech2Text as Speech2TextMaskCTC
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
from src.transforms.WavTransforms import *

def training(e2e, train_loader, optimizer, scheduler, accum_grad):
    e2e.train()
    train_loss = 0.0

    # -- training
    optimizer.zero_grad()
    for batch_idx,(xs_pad, ilens, ys_pad, olens, refs) in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        xs_pad, ilens, ys_pad, olens = xs_pad.to(asr_config.device), ilens.to(asr_config.device), ys_pad.to(asr_config.device), olens.to(asr_config.device)

        loss = e2e(xs_pad, ilens, ys_pad, olens)[0] / accum_grad
        loss.backward()

        # -- updating
        if ((batch_idx+1) % accum_grad == 0) or (batch_idx+1 == len(train_loader)):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss / (len(train_loader) / accum_grad)

def validation(e2e, data_loader):
    e2e.eval()
    data_loss = 0.0
    data_cer = 0.0

    # -- forward pass
    with torch.no_grad():
        for xs_pad, ilens, ys_pad, olens, refs in tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            xs_pad, ilens, ys_pad, olens = xs_pad.to(asr_config.device), ilens.to(asr_config.device), ys_pad.to(asr_config.device), olens.to(asr_config.device)

            loss, stats, weight = e2e(xs_pad, ilens, ys_pad, olens)
            data_loss += loss.item()
            data_cer += stats["cer_ctc"].item() * 100.0

    return round(data_loss / len(data_loader), 3), round(data_cer / len(data_loader), 3)


def inference(output_dir, speech2text, eval_loader, dataset):
    print(f"Decoding {dataset.upper()} dataset:")

    # -- obtaining hypothesis
    dst_dir = os.path.join(output_dir, "inference/")
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, dataset+".inf")
    with open(dst_path, "w") as f:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Acoustic Speech Recognition System based on an End-to-End architecture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--training-dataset", default="", type=str, help="Path to where the training dataset split is")
    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")

    parser.add_argument("--mode", default="both", type=str, help="Choose: 'training', 'inference' or 'both'")
    parser.add_argument("--snr-target", default=9999, type=int, help="A specific signal-to-noise rate when adding noise to the audio waveform.")

    parser.add_argument("--asr-config-file", required=True, type=str, help="Path to a config file that specifies the ASR model architecture")
    parser.add_argument("--load-asr", default="", type=str, help="Path to load a pretrained ASR model")
    parser.add_argument("--lm-config-file", default="", type=str, help="Path to a config file that specifies the LM architecture")
    parser.add_argument("--load-lm", default="", type=str, help="Path to load a pre-trained LM")

    parser.add_argument("--load-modules", nargs='+', default=["entire-e2e"], type=str, help="Choose which parts of the model you want to load: 'entire-e2e', 'frontend', 'encoder' or 'decoder'")
    parser.add_argument("--freeze-modules", nargs='+', default=["no-frozen"], type=str, help="Choose which parts of the model you want to freeze: 'no-frozen', 'frontend', 'encoder' or 'decoder'")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the fine-tuned model and its inference hypothesis")
    parser.add_argument("--output-test", required=True, type=str, help="Name of the file where the hypothesis and results will be write down.")

    args = parser.parse_args()

    # -- configuration architecture details
    asr_config_file = Path(args.asr_config_file)
    with asr_config_file.open("r", encoding="utf-8") as f:
        asr_config = yaml.safe_load(f)
    override_yaml(asr_config, args.yaml_overrides)
    asr_config = argparse.Namespace(**asr_config)

    # -- security checks
    security_checks(asr_config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(asr_config.token_type, asr_config.bpemodel, asr_config.token_list)

    # -- preprocessing data
    train_acoustic_transforms = Compose([
        SpeedRate(sample_rate=16000),
    ])
    eval_acoustic_transforms = Compose([
        AddNoise(noise_path="./src/noise/babble_noise.wav", sample_rate=16000, snr_target=args.snr_target),
    ])

    # -- training
    if args.mode in ["training", "both"]:

        # -- -- building ASR end-to-end system
        e2e = ASRTask.build_model(asr_config).to(
            dtype=getattr(torch, asr_config.dtype),
            device=asr_config.device,
        )
        print(model_summary(e2e))

        # -- -- loading the ASR end-to-end system from a checkpoint
        load_e2e(e2e, args.load_modules, args.load_asr, asr_config.model_conf["ctc_weight"])

        # -- -- freezing modules of the ASR end-to-end system
        freeze_e2e(e2e, args.freeze_modules, asr_config.model_conf["ctc_weight"])

        # -- -- creating dataloaders
        train_loader = get_dataloader(asr_config, dataset_path=args.training_dataset, transforms=train_acoustic_transforms, tokenizer=tokenizer, converter=converter, is_training=True, modality="audio")
        val_loader = get_dataloader(asr_config, dataset_path=args.validation_dataset, transforms=eval_acoustic_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality="audio")
        test_loader = get_dataloader(asr_config, dataset_path=args.test_dataset, transforms=eval_acoustic_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality="audio")

        # -- -- optimizer and scheduler
        optimizer, scheduler = set_optimizer(asr_config, e2e, train_loader)

        # -- -- training process
        print("\nTRAINING PHASE\n")
        val_stats = []
        for epoch in range(1, asr_config.epochs+1):
            train_loss = training(e2e, train_loader, optimizer, scheduler, asr_config.accum_grad)
            val_loss, val_cer = validation(e2e, val_loader)
            test_loss, test_cer = validation(e2e, test_loader)

            print(f"Epoch {epoch}: TRAIN LOSS={train_loss} || VAL LOSS={val_loss} | VAL CER={val_cer}% || TEST LOSS={test_loss} | TEST CER={test_cer}%")
            dst_check_path = save_model(args.output_dir, e2e, str(epoch).zfill(3))
            val_stats.append( (dst_check_path, val_cer) )

        # -- -- computing average model
        save_val_stats(args.output_dir, val_stats)
        sorted_val_stats = sorted(val_stats, key=lambda x: x[1])
        check_paths = [check_path for check_path, cer in sorted_val_stats[:asr_config.average_epochs]]
        average_model(e2e, check_paths)
        save_model(args.output_dir, e2e, "average")

    # -- inference
    if args.mode in ["inference", "both"]:
        print("\nINFERENCE PHASE\n")

        # -- -- building speech-to-text recoginiser
        if asr_config.model == "espnet":
            speech2text = Speech2Text(
                asr_train_config= args.asr_config_file,
                asr_model_file=args.load_asr if args.mode == "inference" else os.path.join(args.output_dir, "models/model_average.pth"),
                lm_train_config=args.lm_config_file if args.lm_config_file != "" else None,
                lm_file=args.load_lm if args.load_lm != "" else None,
                **asr_config.inference_conf,
            )
        elif asr_config.model == "maskctc":
            speech2text = Speech2TextMaskCTC(
                asr_train_config=args.asr_config_file,
                asr_model_file=args.load_asr if args.mode == "inference" else os.path.join(args.output_dir, "models/model_average.pth"),
                token_type=asr_config.token_type,
                **asr_config.inference_conf,
            )

        # -- -- creating validation & test dataloaders
        # val_loader = get_dataloader(asr_config, dataset_path=args.validation_dataset, transforms=eval_acoustic_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality="audio")
        test_loader = get_dataloader(asr_config, dataset_path=args.test_dataset, transforms=eval_acoustic_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality="audio")

        # inference(args, speech2text, val_loader, "val")
        inference(args.output_dir, speech2text, test_loader, args.output_test)
