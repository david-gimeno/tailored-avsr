import yaml
import argparse
from src.utils import *
from pathlib import Path
from src.WavTransforms import *
from src.tasks.asr import ASRTask
from src.VisualTransforms import *
from fvcore.nn import FlopCountAnalysis, flop_count_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Acoustic Speech Recognition System based on an End-to-End architecture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--device", default="cuda:0", type=str, help="Set device")
    parser.add_argument("--modality", default="audio", type=str, help="Indicates the input data modality")
    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--config-file", required=True, type=str, help="Path to a config file that specifies the model architecture")

    parser.add_argument("--yaml-overrides", metavar="CONF:KEY:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    args = parser.parse_args()

    # -- configuration architecture details
    config_file = Path(args.config_file)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    # -- security checks
    security_checks(config)

    # -- setting device
    device = torch.device(args.device)

    # -- building tokenizer and converter
    tokenizer, converter = get_tokenizer_converter(config.token_type, config.bpemodel, config.token_list)

    # -- preprocessing data
    if args.modality == "audio":
        eval_transforms = Compose([
            AddNoise(noise_path="./src/noise/babble_noise.wav", sample_rate=16000),
        ])
    elif args.modality == "video":
        if ("lrs2" in args.validation_dataset.lower()) or ("lrs3" in args.validation_dataset.lower()):
            (mean, std) = (0.421, 0.165)

        eval_transforms = Compose([
            Normalise(0.0, 250.0),
            Normalise(mean, std),
            CenterCrop((88,88)),
        ])

    # -- dataloader
    val_loader = get_dataloader(config, dataset_path=args.validation_dataset, transforms=eval_transforms, tokenizer=tokenizer, converter=converter, is_training=False, modality=args.modality)

    # -- -- building ASR end-to-end system
    e2e = ASRTask.build_model(config) #.to(device)
    xs_pad, ilens, ys_pad, olens, refs = next(iter(val_loader))

    input = (xs_pad, ilens, ys_pad, olens)
    flops = FlopCountAnalysis(e2e, input)
    print(flop_count_table(flops))
    print(flops.total())
