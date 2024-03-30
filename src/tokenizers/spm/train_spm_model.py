import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimating a Byte Pair Encoding Tokenizer based on a Sentence Piece Processor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--split-path", default="./splits/training/speaker-independent/lrs2bbc+lrs3ted.csv", type=str, help="The split which will indicate the transcriptions to read")
    parser.add_argument("--dst-spm-dir", default="./tokenizers/spm/english/LRS2BBC+LRS3TED/", type=str, help="Directory where the tokenizer will be stored")
    parser.add_argument("--spm-name", default="lrs2bbc+lrs3ted_256vocab", type=str, help="Name of the tokenizer")
    parser.add_argument("--vocab-size", default=256, type=int, help="The size of the tokenizer's vocabulary, including special symbols")

    args = parser.parse_args()

    # -- creating directories
    os.makedirs(args.dst_spm_dir, exist_ok=True)
    dst_spm_prefix = os.path.join(args.dst_spm_dir, args.spm_name)
    dst_training_text_path = dst_spm_prefix+".training"
    dst_token_list_path = dst_spm_prefix+".token"

    # -- obtaining split
    transcription_split = pd.read_csv(args.split_path)["transcription_path"].tolist()

    # -- processing database transcriptions defined by the split

    with open(dst_training_text_path, "w") as w:
        for transcription_path in tqdm(transcription_split):
            with open(transcription_path, "r") as r:
                # -- reading database transcriptions
                transcription = r.read().strip().upper().replace("{", "").replace("}","")

                # -- writing text file for training the SentencePiece model
                w.write(transcription + "\n")

    # -- estimating Sentencepiece model
    spm.SentencePieceTrainer.train(
        f"--input={dst_training_text_path} "
        f"--model_prefix={dst_spm_prefix} "
        f"--user_defined_symbols=<blank>,<sos/eos> "
        f"--unk_id=1 "
        f"--bos_id=-1 "
        f"--eos_id=-1 "
        f"--pad_id=-1 "
        f"--vocab_size={args.vocab_size}"
    )

    # -- obtaining token list
    sp = spm.SentencePieceProcessor()
    sp.load(dst_spm_prefix+".model")
    with open(dst_token_list_path, "w") as w:
        for i in range(args.vocab_size):
            w.write(sp.id_to_piece(i) + "\n")
