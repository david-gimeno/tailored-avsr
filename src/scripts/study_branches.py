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

from src.transforms.WavTransforms import *
from src.transforms.VisualTransforms import *

def get_branch_weights(e2e, num_layers, num_groups, data_loader, device, output_dir, output_name):
    e2e.eval()

    stats_df = []
    # -- computing the influence/score of each branch in each layer of a Branchformer-based Encoder
    global_weights = np.zeros(num_layers)
    local_weights = np.zeros(num_layers)
    with torch.no_grad():
        for batch_idx, (xs_pad, ilens, ys_pad, olens, refs) in enumerate(tqdm(data_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET))):
            xs_pad, ilens, ys_pad, olens = xs_pad.to(device), ilens.to(device), ys_pad.to(device), olens.to(device)

            _ = e2e(xs_pad, ilens, ys_pad, olens)[0].mean()

            sample_global_weights = []
            sample_local_weights = []
            if num_groups is None:
                # -- base implementation
                for nlayer in range(num_layers):
                    transcription = refs[0]

                    global_weight = e2e.encoder.encoders[nlayer].weight_global.item()
                    local_weight = e2e.encoder.encoders[nlayer].weight_local.item()

                    sample_global_weights.append(global_weight)
                    sample_local_weights.append(local_weight)

                    stats_df.append( (batch_idx, nlayer, transcription, global_weight, local_weight, len(transcription.split())) )

            else:
                # -- sim-T implementation
                num_layers_per_group = num_layers // num_groups
                for ngroup in range(num_groups):
                    group_global_weights = e2e.encoder.encoders[ngroup].weight_global
                    group_local_weights = e2e.encoder.encoders[ngroup].weight_local

                    for nlayer in range(num_layers_per_group):
                        sample_global_weights.append(group_global_weights[nlayer].item())
                        sample_local_weights.append(group_local_weights[nlayer].item())

            global_weights += np.array(sample_global_weights)
            local_weights += np.array(sample_local_weights)


    avg_global_weights = global_weights / len(data_loader)
    avg_local_weights = local_weights / len(data_loader)

    both_weights = np.moveaxis(np.vstack((avg_global_weights, avg_local_weights)), 0, -1)

    print("GLOBAL:\n", avg_global_weights, avg_global_weights.shape)
    print("\nLOCAL:\n", avg_local_weights, avg_local_weights.shape)
    print("\nHEATMAP:\n", both_weights, both_weights.shape)

    hm = sns.heatmap(both_weights, vmin=0, vmax=1, annot=True, fmt=".4f", cmap="Blues", xticklabels=["Self-Attention Branch", "cgMLP Branch"], yticklabels=["Layer "+str(i) for i in range(num_layers)])
    fig = hm.get_figure()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, output_name))

    stats_df = pd.DataFrame(stats_df, columns=["batch_idx", "nlayer", "transcription", "global_weight", "local_weight", "nwords"])
    stats_df.to_csv(os.path.join(output_dir, output_name.replace(".png", ".csv")), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to study the influence of each branch in each layer of the Branchformer Encoder.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")
    parser.add_argument("--model-config-file", required=True, type=str, help="Path to a config file that specifies the VSR model architecture.")
    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")
    parser.add_argument("--load-model", required=True, type=str, help="Path to load a pretrained VSR model.")
    parser.add_argument("--modality", default="audio", type=str, help="Choose the modality: audio, video, and audiovisual.")
    parser.add_argument("--snr-target", default=9999, type=int, help="Specific signal-to-noise rate when injecting noise to the audio waveform.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory where the heatmap of the influence's branches will be store.")
    parser.add_argument("--output-name", required=True, type=str, help="Name the heatmap PNG file.")

    args = parser.parse_args()

    # -- getting torch device
    device = torch.device("cpu")

    # -- preprocessing data
    if args.modality == "video":
        if "lip-rtve" in args.test_dataset.lower():
            (mean, std) = (0.491, 0.166)
        elif "vlrf" in args.test_dataset.lower():
            (mean, std) = (0.392, 0.142)
        else:
            (mean, std) = (0.421, 0.165)
        fps = 50 if "vlrf" in args.test_dataset.lower() else 25

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

    # -- defining test data loader
    test_loader = get_dataloader(model_config, dataset_path=args.test_dataset, transforms=eval_data_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality=args.modality)

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

    # -- getting brach weights
    num_layers = model_config.encoder_conf["num_blocks"]
    num_groups = model_config.encoder_conf.get("num_groups")

    # output_name = args.test_dataset.split("/")[-1].replace(".csv", ".png")
    get_branch_weights(speech2text.asr_model, num_layers, num_groups, test_loader, device, args.output_dir, args.output_name+".png")
