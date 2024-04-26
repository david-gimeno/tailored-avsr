import torch
import torch.nn as nn
import torch.utils.data as data
from src.datasets import LMDataset

def get_lm_dataloader(config, dataset_path, tokenizer, converter, is_training=True):

    # -- creating dataset
    dataset = LMDataset(
        dataset_path = dataset_path,
        from_dataset_partiton = ".csv" in dataset_path,
    )

    # -- defining dataloader
    dataloader = data.DataLoader(
        dataset = dataset,
        batch_size = config.training_settings['batch_size'] if is_training else 1,
        shuffle = is_training,
        collate_fn = lambda x: lm_data_processing(x, tokenizer, converter, config.model_conf["ignore_id"]),
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
    )

    return dataloader

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
