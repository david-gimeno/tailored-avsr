import os
import torch
import torch.nn as nn
import torch.utils.data as data
from src.datasets import AVSRDataset

def get_dataloader(config, dataset_path, audio_transforms, video_transforms, tokenizer, converter, is_training=True):

    # -- defining dataset
    dataset = AVSRDataset(
        config,
        dataset_path=dataset_path,
        is_training=is_training,
    )

    # -- data processing function
    if config.task == 'asr':
        data_processing_func = asr_data_processing
    elif config.task == 'vsr':
        data_processing_func = vsr_data_processing
    elif config.task == 'avsr':
        data_processing_func = avsr_data_processing
    else:
        raise ValueError(f'unknown task {config.task}')

    # -- defining dataloader
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config.training_settings['batch_size'] if is_training else 1,
        shuffle=is_training,
        collate_fn=lambda x: data_processing_func(x, audio_transforms, video_transforms, tokenizer, converter, config),
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
    )

    return dataloader

def asr_data_processing(data, audio_transforms, video_transforms, tokenizer, converter, config):
    # -- create empty batch
    batch = {'sample_id': [], 'speech': [], 'speech_lengths': [], 'text': [], 'text_lengths': [], 'refs': []} # list(data[0].keys()) + ['audio_ilens', 'y_labels', 'y_olens', 'refs']
    # batch = {data_key: [] for data_key in batch_keys}

    for sample in data:
        batch['sample_id'].append(sample['sample_id'])

        # -- audio preprocessing
        audio = audio_transforms(sample['audio']) if audio_transforms else sample['audio']
        audio = audio.transpose(1,0)
        audio_lengths = audio.shape[0]
        audio = audio[:audio_lengths // 640 * 640, :]

        batch['speech'].append(audio)
        batch['speech_lengths'].append(audio.shape[0])

        # -- transcription preprocessing
        text = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(sample['transcription'])))

        batch['text'].append(text)
        batch['text_lengths'].append(text.shape[0])

        batch['refs'].append(sample['transcription'])

    # -- speech sequence padding
    batch['speech'] = nn.utils.rnn.pad_sequence(batch['speech'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.float32) # -- (#batch, time, channel)
    batch['speech_lengths'] = torch.Tensor(batch['speech_lengths']).type(torch.int64) # -- (#batch,)

    batch['text'] = nn.utils.rnn.pad_sequence(batch['text'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.int64)
    batch['text_lengths'] = torch.Tensor(batch['text_lengths']).type(torch.int64) # -- (#batch,)

    return batch

def vsr_data_processing(data, audio_transforms, video_transforms, tokenizer, converter, config):
    # -- create empty batch
    batch = {'sample_id': [], 'speech': [], 'speech_lengths': [], 'text': [], 'text_lengths': [], 'refs': []}

    for sample in data:
        batch['sample_id'].append(sample['sample_id'])

        # -- video preprocessing
        video = video_transforms(sample['video']) if video_transforms else sample['video']

        batch['speech'].append(video)
        batch['speech_lengths'].append(video.shape[0])

        # -- transcription preprocessing
        text = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(sample['transcription'])))

        batch['text'].append(text)
        batch['text_lengths'].append(text.shape[0])

        batch['refs'].append(sample['transcription'])

    # -- speech sequence padding
    batch['speech'] = nn.utils.rnn.pad_sequence(batch['speech'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.float32) # -- (#batch, time, width, height)
    batch['speech_lengths'] = torch.Tensor(batch['speech_lengths']).type(torch.int64) # -- (#batch,)

    batch['text'] = nn.utils.rnn.pad_sequence(batch['text'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.int64)
    batch['text_lengths'] = torch.Tensor(batch['text_lengths']).type(torch.int64) # -- (#batch,)

    return batch

def avsr_data_processing(data, audio_transforms, video_transforms, tokenizer, converter, config):
    # -- create empty batch
    batch = {'sample_id': [], 'audio': [], 'audio_lengths': [], 'video': [], 'video_lengths': [], 'text': [], 'text_lengths': [], 'refs': []}

    for sample in data:
        batch['sample_id'].append(sample['sample_id'])

        # -- audio preprocessing
        audio = audio_transforms(sample['audio']) if audio_transforms else sample['audio']
        audio = audio.transpose(1,0)
        audio_lengths = audio.shape[0]
        audio = audio[:audio_lengths // 640 * 640, :]

        batch['audio'].append(audio)
        batch['audio_lengths'].append(audio.shape[0])

        # -- video preprocessing
        video = video_transforms(sample['video']) if video_transforms else sample['video']

        batch['video'].append(video)
        batch['video_lengths'].append(video.shape[0])

        # -- transcription preprocessing
        text = torch.Tensor(converter.tokens2ids(tokenizer.text2tokens(sample['transcription'])))

        batch['text'].append(text)
        batch['text_lengths'].append(text.shape[0])

        batch['refs'].append(sample['transcription'])

    # -- speech sequence padding
    batch['audio'] = nn.utils.rnn.pad_sequence(batch['audio'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.float32) # -- (#batch, time, channel)
    batch['audio_lengths'] = torch.Tensor(batch['audio_lengths']).type(torch.int64) # -- (#batch,)

    batch['video'] = nn.utils.rnn.pad_sequence(batch['video'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.float32) # -- (#batch, time, width, height)
    batch['video_lengths'] = torch.Tensor(batch['video_lengths']).type(torch.int64) # -- (#batch,)

    batch['text'] = nn.utils.rnn.pad_sequence(batch['text'], padding_value=config.model_conf['ignore_id'], batch_first=True).type(torch.int64)
    batch['text_lengths'] = torch.Tensor(batch['text_lengths']).type(torch.int64) # -- (#batch,)

    return batch
