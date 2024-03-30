#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.tasks.avsr import AVSRTask
from src.evaluation.Tasas import computeTasas
from espnet2.torch_utils.model_summary import model_summary
from src.bin.avsr_inference import Speech2Text
from src.bin.avsr_inference_maskctc import Speech2Text as Speech2TextMaskCTC


import os
import sys
import yaml
import argparse
from tqdm import tqdm
from colorama import Fore
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.utils import *
from src.transforms.WavTransforms import *
from torchvision import transforms
from src.transforms.VisualTransforms import *

def training(e2e, train_loader, optimizer, scheduler, accum_grad, scaler=None):
    e2e.train()
    train_loss = 0.0

    # -- training
    optimizer.zero_grad()
    for batch_idx,(audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens, refs) in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens = audio_pad.to(avsr_config.device), audio_ilens.to(avsr_config.device), video_pad.to(avsr_config.device), video_ilens.to(avsr_config.device), ys_pad.to(avsr_config.device), olens.to(avsr_config.device)

        with autocast(scaler is not None, dtype=torch.bfloat16):
            loss = e2e(audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens)[0] / accum_grad

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # -- updating
        if ((batch_idx+1) % accum_grad == 0) or (batch_idx+1 == len(train_loader)):
            if scaler is not None:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
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
        for audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens, refs in tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens = audio_pad.to(avsr_config.device), audio_ilens.to(avsr_config.device), video_pad.to(avsr_config.device), video_ilens.to(avsr_config.device), ys_pad.to(avsr_config.device), olens.to(avsr_config.device)

            loss, stats, weight = e2e(audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens)
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
            for audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens, refs in tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.YELLOW, Fore.RESET)):

                if args.mask == "audio":
                    audio_pad = audio_pad * 0
                if args.mask == "video":
                    video_pad = video_pad * 0

                result = speech2text(torch.squeeze(audio_pad, 0), torch.squeeze(video_pad, 0))
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
    parser = argparse.ArgumentParser(description="Automatic Audio-Visual Speech Recognition System.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--training-dataset", default="", type=str, help="Path to where the training dataset split is")
    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")

    parser.add_argument("--mode", default="both", type=str, help="Choose: 'training', 'inference' or 'both'")
    parser.add_argument("--mask", default="none", type=str, help="Choose: 'audio', 'video' or 'none'")
    parser.add_argument("--snr-target", default=9999, type=int, help="A specific signal-to-noise rate when adding noise to the audio waveform.")

    parser.add_argument("--avsr-config-file", required=True, type=str, help="Path to a config file that specifies the AVSR model architecture")
    parser.add_argument("--load-avsr", default="", type=str, help="Path to load a pretrained AVSR model")
    parser.add_argument("--lm-config-file", default="", type=str, help="Path to a config file that specifies the LM architecture")
    parser.add_argument("--load-lm", default="", type=str, help="Path to load a pre-trained LM")

    parser.add_argument("--load-modules", nargs='+', default=["entire-e2e"], type=str, help="Choose which parts of the model you want to load: 'entire-e2e', 'frontend', 'encoder' or 'decoder'")
    parser.add_argument("--freeze-modules", nargs='+', default=["no-frozen"], type=str, help="Choose which parts of the model you want to freeze: 'no-frozen', 'frontend', 'encoder' or 'decoder'")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the fine-tuned model and its inference hypothesis")
    parser.add_argument("--output-test", required=True, type=str, help="Name of the file where the hypothesis and results will be write down.")

    args = parser.parse_args()

    # -- configuration architecture details
    avsr_config_file = Path(args.avsr_config_file)
    with avsr_config_file.open("r", encoding="utf-8") as f:
        avsr_config = yaml.safe_load(f)
    override_yaml(avsr_config, args.yaml_overrides)
    avsr_config = argparse.Namespace(**avsr_config)

    # -- security checks
    security_checks(avsr_config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(avsr_config.token_type, avsr_config.bpemodel, avsr_config.token_list)

    # -- acoustic preprocessing data
    train_acoustic_transforms = Compose([
        SpeedRate(sample_rate=16000),
    ])
    eval_acoustic_transforms = Compose([
        AddNoise(noise_path="./src/noise/babble_noise.wav", sample_rate=16000, snr_target=args.snr_target),
    ])

    # -- visual preprocessing data
    fps = 25
    (mean, std) = (0.421, 0.165)

    train_visual_transforms = transforms.Compose([
        Normalise(0.0, 250.0),
        Normalise(mean, std),
        TimeMasking(fps=fps, max_seconds=0.4),
        transforms.RandomCrop((88,88)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    eval_visual_transforms = Compose([
        Normalise(0.0, 250.0),
        Normalise(mean, std),
        CenterCrop((88,88)),
    ])

    # -- training
    if args.mode in ["training", "both"]:

        # -- -- building AVSR end-to-end system
        e2e = AVSRTask.build_model(avsr_config).to(
            dtype=getattr(torch, avsr_config.dtype),
            device=avsr_config.device,
        )
        print(model_summary(e2e))

        # -- -- loading the AVSR end-to-end system from a checkpoint
        load_e2e(e2e, args.load_modules, args.load_avsr, avsr_config.model_conf["ctc_weight"])

        # -- -- freezing modules of the AVSR end-to-end system
        freeze_e2e(e2e, args.freeze_modules, avsr_config.model_conf["ctc_weight"])

        # -- -- creating dataloaders
        train_loader = get_audiovisual_dataloader(avsr_config, dataset_path=args.training_dataset, acoustic_transforms=train_acoustic_transforms, visual_transforms=train_visual_transforms, tokenizer=tokenizer, converter=converter, is_training=True)
        val_loader = get_audiovisual_dataloader(avsr_config, dataset_path=args.validation_dataset, acoustic_transforms=eval_acoustic_transforms, visual_transforms=eval_visual_transforms, tokenizer=tokenizer, converter=converter, is_training=False)
        test_loader = get_audiovisual_dataloader(avsr_config, dataset_path=args.test_dataset, acoustic_transforms=eval_acoustic_transforms, visual_transforms=eval_visual_transforms, tokenizer=tokenizer, converter=converter, is_training=False)

        # -- -- optimizer and scheduler
        optimizer, scheduler = set_optimizer(avsr_config, e2e, train_loader)

        # -- -- training process
        print("\nTRAINING PHASE\n")
        val_stats = []
        scaler = GradScaler if avsr_config.use_amp else None

        # e = 10
        # spe = 1792
        # for e in range(e*spe):
        #     optimizer.step()

        for epoch in range(1, avsr_config.epochs+1):
            train_loss = training(e2e, train_loader, optimizer, scheduler, avsr_config.accum_grad, scaler)
            val_loss, val_cer = validation(e2e, val_loader)
            test_loss, test_cer = validation(e2e, test_loader)

            print(f"Epoch {epoch}: TRAIN LOSS={train_loss} || VAL LOSS={val_loss} | VAL CER={val_cer}% || TEST LOSS={test_loss} | TEST CER={test_cer}%")
            dst_check_path = save_model(args.output_dir, e2e, str(epoch).zfill(3))
            val_stats.append( (dst_check_path, val_cer) )

        # -- -- computing average model
        save_val_stats(args.output_dir, val_stats)
        sorted_val_stats = sorted(val_stats, key=lambda x: x[1])
        check_paths = [check_path for check_path, cer in sorted_val_stats[:avsr_config.average_epochs]]
        average_model(e2e, check_paths)
        save_model(args.output_dir, e2e, "average")

    # -- inference
    if args.mode in ["inference", "both"]:
        print("\nINFERENCE PHASE\n")

        # -- -- building speech-to-text recoginiser
        if avsr_config.model == "espnet":
            speech2text = Speech2Text(
                asr_train_config= args.avsr_config_file,
                asr_model_file=args.load_avsr if args.load_avsr != "" else os.path.join(args.output_dir, "models/model_average.pth"),
                lm_train_config=args.lm_config_file if args.lm_config_file != "" else None,
                lm_file=args.load_lm if args.load_lm != "" else None,
                **avsr_config.inference_conf,
            )
        elif avsr_config.model == "maskctc":
            speech2text = Speech2TextMaskCTC(
                asr_train_config=args.avsr_config_file,
                asr_model_file=args.load_avsr if args.load_avsr != "" else os.path.join(args.output_dir, "models/model_average.pth"),
                token_type=avsr_config.token_type,
                **avsr_config.inference_conf,
            )

        # -- -- creating validation & test dataloaders
        # val_loader = get_audiovisual_dataloader(avsr_config, dataset_path=args.validation_dataset, acoustic_transforms=eval_acoustic_transforms, visual_transforms=eval_visual_transforms, tokenizer=tokenizer, converter=converter, is_training=False)
        test_loader = get_audiovisual_dataloader(avsr_config, dataset_path=args.test_dataset, acoustic_transforms=eval_acoustic_transforms, visual_transforms=eval_visual_transforms, tokenizer=tokenizer, converter=converter, is_training=False)

        # inference(args.output_dir, speech2text, val_loader, "val")
        inference(args.output_dir, speech2text, test_loader, args.output_test)

