import os
import math
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import OrderedDict

from src.datasets.MyAudioDataset import MyAudioDataset
from src.datasets.MyVideoDataset import MyVideoDataset
from src.datasets.MyAudiovisualDataset import MyAudiovisualDataset
from src.datasets.MyTextDataset import MyTextDataset

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

from src.schedulers.noam import get_noam_scheduler

def override_yaml(yaml_config, to_override):
    if to_override is not None:
        for new_setting in to_override:
            if new_setting.count(":") == 1:
                key, value = new_setting.split(":")
                value_type_func = type(yaml_config[key])
                if value_type_func == bool:
                    yaml_config[key] = value == "true"
                else:
                    yaml_config[key] = value_type_func(value)

            elif new_setting.count(":") == 2:
                conf, key, value = new_setting.split(":")
                value_type_func = type(yaml_config[conf][key])
                if value_type_func == bool:
                    yaml_config[conf][key] = value == "true"
                else:
                    yaml_config[conf][key] = value_type_func(value)

    return yaml_config

def data_processing(data, transforms, tokenizer, converter, ignore_id, modality="video"):
    x_speech = []
    x_ilens = []
    y_labels = []
    y_olens = []
    refs = []

    for speech, transcription in data:
        speech = transforms(speech) if transforms else speech

        if modality == "audio":
            speech = speech.transpose(1,0)
            T = speech.shape[0]
            speech = speech[:T // 640 * 640, :]
            x_speech.append(speech)
            x_ilens.append(speech.shape[0])
        elif modality == "video":
            x_speech.append(speech)
            x_ilens.append(speech.shape[0])

        label = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(transcription)))
        y_labels.append(label)
        y_olens.append(label.shape[0])

        refs.append(transcription)

    # -- audio: (#batch, time, channel) || video: (#batch, time, width, height)
    x_speech = nn.utils.rnn.pad_sequence(x_speech, padding_value=ignore_id, batch_first=True).type(torch.float32)
    x_ilens = torch.Tensor(x_ilens).type(torch.int64) # -- (#batch,)
    # -- (#batch, time)
    y_labels = nn.utils.rnn.pad_sequence(y_labels, padding_value=ignore_id, batch_first=True).type(torch.int64)
    y_olens = torch.Tensor(y_olens).type(torch.int64) # -- (#batch,)

    return x_speech, x_ilens, y_labels, y_olens, refs

def audiovisual_data_processing(data, acoustic_transforms, visual_transforms, tokenizer, converter, ignore_id):
    audio_speech = []
    audio_ilens = []
    video_speech = []
    video_ilens = []
    y_labels = []
    y_olens = []
    refs = []

    for audio, video, transcription in data:
        # -- audio preprocessing
        audio = acoustic_transforms(audio) if acoustic_transforms else audio
        audio = audio.transpose(1,0)
        Ta = audio.shape[0]
        audio = audio[:Ta // 640 * 640, :]
        audio_speech.append(audio)
        audio_ilens.append(audio.shape[0])

        # -- video preprocessing
        video = visual_transforms(video) if visual_transforms else video
        video_speech.append(video)
        video_ilens.append(video.shape[0])

        label = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(transcription)))
        y_labels.append(label)
        y_olens.append(label.shape[0])

        refs.append(transcription)

    # -- audio: (#batch, time, channel)
    audio_speech = nn.utils.rnn.pad_sequence(audio_speech, padding_value=ignore_id, batch_first=True).type(torch.float32)
    audio_ilens = torch.Tensor(audio_ilens).type(torch.int64) # -- (#batch,)
    # -- video: (#batch, time, width, height)
    video_speech = nn.utils.rnn.pad_sequence(video_speech, padding_value=ignore_id, batch_first=True).type(torch.float32)
    video_ilens = torch.Tensor(video_ilens).type(torch.int64) # -- (#batch,)
    # -- (#batch, time)
    y_labels = nn.utils.rnn.pad_sequence(y_labels, padding_value=ignore_id, batch_first=True).type(torch.int64)
    y_olens = torch.Tensor(y_olens).type(torch.int64) # -- (#batch,)

    return audio_speech, audio_ilens, video_speech, video_ilens, y_labels, y_olens, refs

def lm_data_processing(data, tokenizer, converter, ignore_id):
    x_tokens = []
    x_ilens = []
    refs = []

    for text in data:
        token_ids = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(text)))
        x_tokens.append(token_ids)
        x_ilens.append(token_ids.shape[0])

        refs.append(text)

    x_tokens = nn.utils.rnn.pad_sequence(x_tokens, padding_value=ignore_id, batch_first=True).type(torch.int64)
    x_ilens = torch.Tensor(x_ilens).type(torch.int64) # -- (#batch,)

    return x_tokens, x_ilens, refs

def get_tokenizer_converter(token_type, bpemodel, token_list):
    if token_type is None:
        tokenizer = None
    elif (
        token_type == "bpe"
        or token_type == "hugging_face"
        or "whisper" in token_type
    ):
        if bpemodel is not None:
            tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        else:
            tokenizer = None
    else:
        tokenizer = build_tokenizer(token_type=token_type)

    if bpemodel not in ["whisper_en", "whisper_multilingual"]:
        converter = TokenIDConverter(token_list=token_list)
    else:
        converter = OpenAIWhisperTokenIDConverter(model_type=bpemodel)

    return tokenizer, converter

def get_dataloader(config, dataset_path, transforms, tokenizer, converter, is_training=True, modality="audio"):
    kwargs = {"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {}

    # -- creating dataset
    if modality == "audio":
        dataset_class = MyAudioDataset
    elif modality == "video":
        dataset_class = MyVideoDataset
    elif modality == "audiovisual":
        dataset_class = MyAudiovisualDataset

    dataset = dataset_class(
        dataset_path=dataset_path,
        nframes=config.nframes,
        is_training=is_training,
    )

    # -- defining dataloader
    if not is_training:
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, transforms, tokenizer, converter, config.model_conf["ignore_id"], modality=modality),
            **kwargs,
        )
    else:
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: data_processing(x, transforms, tokenizer, converter, config.model_conf["ignore_id"], modality=modality),
            **kwargs,
        )

    return dataloader

def get_audiovisual_dataloader(config, dataset_path, acoustic_transforms, visual_transforms, tokenizer, converter, is_training=True):
    kwargs = {"num_workers": 0, "pin_memory": True} if torch.cuda.is_available() else {}

    # -- creating dataset
    dataset = MyAudiovisualDataset(
        dataset_path=dataset_path,
        nframes=config.nframes,
        is_training=is_training,
    )

    # -- defining dataloader
    if not is_training:
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: audiovisual_data_processing(x, acoustic_transforms, visual_transforms, tokenizer, converter, config.model_conf["ignore_id"]),
            **kwargs,
        )
    else:
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: audiovisual_data_processing(x, acoustic_transforms, visual_transforms, tokenizer, converter, config.model_conf["ignore_id"]),
            **kwargs,
        )

    return dataloader

def get_lm_dataloader(config, dataset_path, tokenizer, converter, is_training=True):
    kwargs = {"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {}

    # -- creating dataset
    dataset_class = MyTextDataset

    dataset = dataset_class(
        dataset_path = dataset_path,
        from_dataset_partiton = ".csv" in dataset_path,
    )

    # -- defining dataloader
    if not is_training:
        dataloader = data.DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = False,
            collate_fn = lambda x: lm_data_processing(x, tokenizer, converter, config.model_conf["ignore_id"]),
            **kwargs,
        )
    else:
        dataloader = data.DataLoader(
            dataset = dataset,
            batch_size = config.batch_size,
            shuffle = False,
            collate_fn = lambda x: lm_data_processing(x, tokenizer, converter, config.model_conf["ignore_id"]),
            **kwargs,
        )

    return dataloader


def load_frontend_lrw(e2e, checkpoint, module_name):
    frontend_lrw = OrderedDict()
    for key in checkpoint.keys():
        if "tcn_trunk" not in key:
            if ("trunk" in key) or ("frontend3D" in key):
                frontend_lrw[key] = checkpoint[key]

    if module_name == "frontend":
        e2e.frontend.load_state_dict(frontend_lrw)
    elif module_name == "visual_frontend":
        e2e.visual_frontend.load_state_dict(frontend_lrw)

def load_module(e2e, module, checkpoint, ctc_weight):
    # -- creating the module's checkpoint
    module_checkpoint = OrderedDict()
    for key in checkpoint.keys():
        if module+"." in key:
            new_key = key.replace(module+".", "")
            module_checkpoint[new_key] = checkpoint[key]

    # -- loading the chosen module
    if module == "frontend":
        e2e.frontend.load_state_dict(module_checkpoint)

    if module == "encoder":
        e2e.encoder.load_state_dict(module_checkpoint)

    if module == "decoder":
        if ctc_weight < 1.0:
            e2e.decoder.load_state_dict(module_checkpoint)
        else:
            raise RuntimeError("The end-to-end model does not have an Attention-based decoding branch!")

    if module == "ctc":
        if ctc_weight > 0.0:
            e2e.ctc.load_state_dict(module_checkpoint)
        else:
            raise RuntimeError("The end-to-end model does not have a CTC-based decoding branch!")

def load_e2e(e2e, modules, checkpoint_path, ctc_weight):
    if checkpoint_path != "":
        checkpoint = torch.load(checkpoint_path)
        if "entire-e2e" not in modules:
            for module in modules:
                if ("LRW" in checkpoint_path):
                    assert module in ["frontend", "visual_frontend"], "When loading from the LRW model, it is only possible loading the frontend."
                    print(f"Loading pre-trained visual frontend from {checkpoint_path}")
                    load_frontend_lrw(e2e, checkpoint, module)
                else:
                    print(f"Loading pre-trained {module} from {checkpoint_path}.")
                    load_module(e2e, module, checkpoint, ctc_weight)
        else:
            print(f"Loading the entire E2E system from {checkpoint_path}")
            e2e.load_state_dict(checkpoint, strict=False)
            # if (ctc_weight > 0.0) and (ctc_weight < 1.0):
            #      e2e.load_state_dict(checkpoint)
            # else: # If there is no Attention- or CTC-based branch
            #      e2e.load_state_dict(checkpoint, strict=False)
            #      print(f"The end-to-end model was pre-train but there is a mismatch. It is probably missing the Attention- or the CTC-based decoding branch.")
    else:
        print(f"Training the end-to-end model from scracth!")

def average_e2e(config, output_dir, e2e):
    if config.average_epochs > 0:
        print(f"Computing the average of the last {config.average_epochs} epochs...")
        checkpoint_paths = []
        first_epoch = (config.epochs - config.average_epochs) + 1
        for i in range(first_epoch, config.epochs+1):
            checkpoint_path = os.path.join(output_dir, "models/model_" + str(i).zfill(3) + ".pth")
            checkpoint_paths.append(checkpoint_path)
        average_model(e2e, checkpoint_paths)
        save_model(output_dir, e2e, "average")

def average_model(e2e, checkpoint_paths):
    """
      Code based on the implentation publicly released by FairSeq.
        https://github.com/facebookresearch/fairseq/blob/main/scripts/average_checkpoints.py
    """
    average_model = {}
    for checkpoint_path in checkpoint_paths:
        model = torch.load(checkpoint_path)

        model_keys = list(model.keys())
        for k in model_keys:
            p = model[k]
            if average_model.get(k) is None:
                average_model[k] = p.clone()
            else:
                average_model[k] += p.clone()

    nmodels = len(checkpoint_paths)
    for k, v in average_model.items():
        average_model[k] = torch.div(v, nmodels)

    e2e.load_state_dict(average_model)

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def freeze_e2e(e2e, modules, mtlalpha):
    if "no-frozen" not in modules:
        for module in modules:
            if module == "frontend":
                for param in e2e.frontend.parameters():
                    param.requires_grad = False
                print("The Frontend is frozen!!")
            elif module == "encoder":
                for param in e2e.encoder.parameters():
                    param.requires_grad = False
                print("The Encoder is frozen!!")
            elif module == "decoder":
                if mtlalpha < 1.0:
                    for param in e2e.decoder.parameters():
                        param.requires_grad = False
                    print("The Attention-based Decoder is frozen!!")
                else:
                    raise RuntimeError("The end-to-end model does not have a Attention-based decoding branch!")
            elif module == "ctc":
                if mtlalpha > 0.0:
                    for param in e2e.ctc.parameters():
                        param.requieres_grad = False
                    print("The CTC-based Decoder is frozen!!")
                else:
                    raise RuntimeError("The end-to-end model does not have a CTC-based decoding branch!")
    else:
        print("The entire E2E system will be trained")

def set_optimizer(config, e2e, train_loader):
    optimizer = None
    scheduler = None

    # -- computing steps per epoch considering the accumulation gradient
    if config.accum_grad != 0:
        steps_per_epoch = math.ceil(len(train_loader) / config.accum_grad)
    else:
        steps_per_epoch = len(train_loader)
    print(f"\nTrainLoader length with a batch size of {config.batch_size}: {len(train_loader)} batches")
    print(f"Accumulation Gradient during {config.accum_grad} steps => Simulated Batch Size of {config.batch_size * max(1, config.accum_grad)} samples")
    print(f"Computed steps per epoch: {steps_per_epoch}")

    ## -- defining optimizer and scheduler
    if config.scheduler != "noam":
        print(f"Setting {config.optimizer} optimizer with {config.scheduler} scheduler.")
        if config.optimizer == "adamw":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, e2e.parameters()), config.learning_rate)
        elif config.optimizer == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, e2e.parameters()), config.learning_rate, betas=(0.9,0.98), eps=10e-09)

    if config.scheduler == "noam":
        print(f"Setting {config.scheduler} optimizer-scheduler.")
        optimizer = get_noam_scheduler(
            e2e.parameters(),
            config.noam_factor,
            config.encoder_conf["output_size"],
            config.warmup_steps,
        )

    elif config.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=config.learning_rate,
                                                  steps_per_epoch=steps_per_epoch,
                                                  epochs=config.epochs,
                                                  anneal_strategy="linear")
    else:
        raise RuntimeError("The scheduler should be specified as 'noam' or 'onecycle'")

    return optimizer, scheduler

def save_model(output_dir, model, suffix):
    dst_root = output_dir + "/models/"

    os.makedirs(dst_root, exist_ok=True)
    dst_path = dst_root + "/model_" + suffix + ".pth"
    print(f"Saving model in {dst_path} ...")
    torch.save(model.state_dict(), dst_path)

    return dst_path

def save_val_stats(output_dir, val_stats):
    dst_path = os.path.join(output_dir, "val_stats.csv")
    df = pd.DataFrame(val_stats, columns=["model_check_path", "cer"])
    df.to_csv(dst_path)

def save_optimizer(args, optimizer, epoch):
    dst_root = args.output_dir + "/optimizer/"

    os.makedirs(dst_root, exist_ok=True)
    dst_path = dst_root + "/optimizer_" + str(epoch).zfill(3) + ".pth"
    print(f"Saving optimizer in {dst_path} ...")
    torch.save(optimizer.state_dict(), dst_path)

def security_checks(config):
    if (config.average_epochs <= 0) or (config.average_epochs > config.epochs):
        raise RuntimeError(
            f"The number of epochs to compute an average model should be a value between 1 and the number of training epochs. You specified (average-epochs, training-epochs): {config.average_epochs},{config.epochs}"
        )
