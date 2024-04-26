#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.tasks import ASRTask
from src.tasks import AVSRTask
from src.evaluation import compute_bootstrap_wer
from espnet2.torch_utils.model_summary import model_summary
from src.inference import ASR2Text, ASR2TextMaskCTC, AVSR2Text, AVSR2TextMaskCTC


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
from src.transforms import *
from torchvision import transforms

def training(e2e, train_loader, optimizer, scheduler, accum_grad, scaler=None):
    e2e.train()
    train_loss = 0.0

    # -- training
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

        loss = e2e(**batch)[0] / config.training_settings['accum_grad']

        # -- forward
        # if config.task == 'asr':
        #     loss = e2e(batch['audio'], batch['audio_ilens'], batch['y_labels'], batch['y_olens'])[0] / config.training_settings['accum_grad']
        # if config.task == 'vsr':
        #     loss = e2e(batch['video'], batch['video_ilens'], batch['y_labels'], batch['y_olens'])[0] / config.training_settings['accum_grad']
        # if config.task == 'avsr':
        #     loss = e2e(batch['audio'], batch['audio_ilens'], batch['video'], batch['video_ilens'], batch['y_labels'], batch['y_olens'])[0] / config.training_settings['accum_grad']

        # -- backward
        loss.backward()

        # -- update
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
        for batch in tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

            loss, stats, weight = e2e(**batch)

            # if config.task == 'asr':
            #     loss, stats, weights = e2e(batch['audio'], batch['audio_ilens'], batch['y_labels'], batch['y_olens'])
            # if config.task == 'vsr':
            #     loss, stats, weights = e2e(batch['video'], batch['video_ilens'], batch['y_labels'], batch['y_olens'])
            # if config.task == 'avsr':
            #     loss, stats, weight = e2e(batch['audio'], batch['audio_ilens'], batch['video'], batch['video_ilens'], batch['y_labels'], batch['y_olens'])

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
            for batch in tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.YELLOW, Fore.RESET)):

                if config.task in ['asr', 'vsr']:
                    result = speech2text(torch.squeeze(batch['speech'], 0))
                elif config.task in ['avsr']:
                    batch['audio'] = batch['audio'] * 0 if args.mask == 'audio' else batch['audio']
                    batch['video'] = batch['video'] * 0 if args.mask == 'video' else batch['video']

                    result = speech2text(torch.squeeze(batch['audio'], 0), torch.squeeze(batch['video'], 0))

                # -- dumping results
                hyp = result[0][0]
                f.write(batch['refs'][0].strip() + "#" + hyp.strip() + "\n")

    # -- computing WER
    wer, cer, ci_wer, ci_cer = compute_bootstrap_wer(dst_path)
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
    parser.add_argument("--noise", default="./src/noise/babble_noise.wav", type=str, help="Path to .wav file of noise")

    parser.add_argument("--config-file", required=True, type=str, help="Path to a config file that specifies the AVSR model architecture")
    parser.add_argument("--load-checkpoint", default="", type=str, help="Path to load a pretrained AVSR model")
    parser.add_argument("--lm-config-file", default="", type=str, help="Path to a config file that specifies the LM architecture")
    parser.add_argument("--load-lm", default="", type=str, help="Path to load a pre-trained LM")

    parser.add_argument("--load-modules", nargs='+', default=["entire-e2e"], type=str, help="Choose which parts of the model you want to load: 'entire-e2e', 'frontend', 'encoder' or 'decoder'")
    parser.add_argument("--freeze-modules", nargs='+', default=["no-frozen"], type=str, help="Choose which parts of the model you want to freeze: 'no-frozen', 'frontend', 'encoder' or 'decoder'")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the fine-tuned model and its inference hypothesis")
    parser.add_argument("--output-name", required=True, type=str, help="Name of the file where the hypothesis and results will be write down.")

    args = parser.parse_args()

    # -- configuration architecture details
    config_file = Path(args.config_file)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    # -- security checks
    security_checks(config)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(config)

    # -- audio preprocessing
    train_audio_transforms = Compose([
        SpeedRate(sample_rate=16000),
    ])
    eval_audio_transforms = Compose([
        AddNoise(noise_path=args.noise, sample_rate=16000, snr_target=args.snr_target),
    ])

    # -- video preprocessing
    fps = 25
    (mean, std) = (0.421, 0.165)

    train_video_transforms = transforms.Compose([
        Normalise(0.0, 250.0),
        Normalise(mean, std),
        TimeMasking(fps=fps, max_seconds=0.4),
        transforms.RandomCrop((88,88)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    eval_video_transforms = Compose([
        Normalise(0.0, 250.0),
        Normalise(mean, std),
        CenterCrop((88,88)),
    ])

    # -- training
    if args.mode in ["training", "both"]:

        # -- -- building AVSR end-to-end system
        task_class = AVSRTask if config.task == 'avsr' else ASRTask
        e2e = task_class.build_model(config).to(
            dtype=getattr(torch, config.dtype),
            device=config.device,
        )
        print(model_summary(e2e))

        # -- -- loading the AVSR end-to-end system from a checkpoint
        load_e2e(e2e, args.load_modules, args.load_checkpoint, config.model_conf["ctc_weight"])

        # -- -- freezing modules of the AVSR end-to-end system
        freeze_e2e(e2e, args.freeze_modules, config.model_conf["ctc_weight"])

        # -- -- creating dataloaders
        train_loader = get_dataloader(config, dataset_path=args.training_dataset, audio_transforms=train_audio_transforms, video_transforms=train_video_transforms, tokenizer=tokenizer, converter=converter, is_training=True)
        val_loader = get_dataloader(config, dataset_path=args.validation_dataset, audio_transforms=eval_audio_transforms, video_transforms=eval_video_transforms, tokenizer=tokenizer, converter=converter, is_training=False)
        test_loader = get_dataloader(config, dataset_path=args.test_dataset, audio_transforms=eval_audio_transforms, video_transforms=eval_video_transforms, tokenizer=tokenizer, converter=converter, is_training=False)

        # -- -- optimizer and scheduler
        optimizer, scheduler = set_optimizer(config, e2e, train_loader)

        # -- -- training process

        # e = 10
        # spe = 1792
        # for e in range(e*spe):
        #     optimizer.step()

        val_stats = []
        print("\nTRAINING PHASE\n")
        scaler = GradScaler if config.training_settings['use_amp'] else None
        for epoch in range(1, config.training_settings['epochs']+1):
            train_loss = training(e2e, train_loader, optimizer, scheduler, config.training_settings['accum_grad'], scaler)
            val_loss, val_cer = validation(e2e, val_loader)
            test_loss, test_cer = validation(e2e, test_loader)

            print(f"Epoch {epoch}: TRAIN LOSS={train_loss} || VAL LOSS={val_loss} | VAL CER={val_cer}% || TEST LOSS={test_loss} | TEST CER={test_cer}%")
            dst_check_path = save_model(args.output_dir, e2e, str(epoch).zfill(3))
            val_stats.append( (dst_check_path, val_cer) )

        # -- -- computing average model
        save_val_stats(args.output_dir, val_stats)
        sorted_val_stats = sorted(val_stats, key=lambda x: x[1])
        check_paths = [check_path for check_path, cer in sorted_val_stats[:config.training_settings['average_epochs']]]
        average_model(e2e, check_paths)
        save_model(args.output_dir, e2e, "average")

    # -- inference
    if args.mode in ["inference", "both"]:
        print("\nINFERENCE PHASE\n")

        # -- -- building speech-to-text recoginiser
        speech2text = build_speech2text(args, config)

        # -- -- creating validation & test dataloaders
        eval_loader = get_dataloader(config, dataset_path=args.test_dataset, audio_transforms=eval_audio_transforms, video_transforms=eval_video_transforms, tokenizer=tokenizer, converter=converter, is_training=False)
        inference(args.output_dir, speech2text, eval_loader, args.output_name)

