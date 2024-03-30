#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.bin.avsr_inference import Speech2Text
from src.bin.avsr_inference_maskctc import Speech2Text as Speech2TextMaskCTC

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

from src.transforms.WavTransforms import *
from src.transforms.VisualTransforms import *

def get_modality_weights(e2e, data_loader, device):
    e2e.eval()

    stats_df = []
    # -- computing the influence/score of each modality for the adaptive audio-visual fusion module

    visual_weights = []
    acoustic_weights = []
    with torch.no_grad():
        for batch_idx, (audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens, refs) in enumerate(tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET))):
            audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens = audio_pad.to(device), audio_ilens.to(device), video_pad.to(device), video_ilens.to(device), ys_pad.to(device), olens.to(device)

            _ = e2e(audio_pad, audio_ilens, video_pad, video_ilens, ys_pad, olens)[0].mean()


            visual_weights.append( e2e.audiovisual_fusion.visual_weight.item() )
            acoustic_weights.append( e2e.audiovisual_fusion.acoustic_weight.item() )

    avg_acoustic_weight = np.array(acoustic_weights).mean()
    avg_visual_weight = np.array(visual_weights).mean()

    print("acoustic:\n", avg_acoustic_weight)
    print("\nvisual:\n", avg_visual_weight)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to study the influence of each branch in each layer of the Branchformer Encoder.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")
    parser.add_argument("--model-config-file", required=True, type=str, help="Path to a config file that specifies the VSR model architecture.")
    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")
    parser.add_argument("--load-model", required=True, type=str, help="Path to load a pretrained VSR model.")
    parser.add_argument("--snr-target", default=9999, type=int, help="Specific signal-to-noise rate when injecting noise to the audio waveform.")

    args = parser.parse_args()

    # -- getting torch device
    device = torch.device("cpu")

    # -- preprocessing data
    if "lip-rtve" in args.test_dataset.lower():
        (mean, std) = (0.491, 0.166)
    elif "vlrf" in args.test_dataset.lower():
        (mean, std) = (0.392, 0.142)
    else:
        (mean, std) = (0.421, 0.165)
    fps = 50 if "vlrf" in args.test_dataset.lower() else 25

    visual_eval_data_transforms = Compose([
        Normalise(0.0, 250.0),
        Normalise(mean, std),
        CenterCrop((88,88)),
    ])

    acoustic_eval_data_transforms = Compose([
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

    # -- defining test data loader
    test_loader = get_audiovisual_dataloader(model_config, dataset_path=args.test_dataset, acoustic_transforms=acoustic_eval_data_transforms, visual_transforms=visual_eval_data_transforms, tokenizer=tokenizer, converter=converter, is_training=False)

    # -- loading end-to-end speech recogniser
    if model_config.model == "espnet":
        speech2text = Speech2Text(
            asr_train_config= args.model_config_file,
            asr_model_file=args.load_model,
            lm_train_config=None,
            lm_file=None,
            **model_config.inference_conf,
        )
    elif model_config.model == "maskctc":
        speech2text = Speech2TextMaskCTC(
            asr_train_config=args.model_config_file,
            asr_model_file=args.load_model,
            token_type=model_config.token_type,
            bpemodel=model_config.bpemodel,
            **model_config.inference_conf,
        )

    get_modality_weights(speech2text.asr_model, test_loader, device)
