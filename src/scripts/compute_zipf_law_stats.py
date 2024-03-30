import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unidecode import unidecode

def compute_zipf_law(database, scenario, datasets, enc, delimiter):
    splits_dir = "../data/" + database + "/splits/" + scenario + "/"
    transcriptions_path = "../data/" + database + "/transcriptions/"

    cont = 0
    word_freq = {}
    for dataset_name in datasets:
        dataset_path = os.path.join(splits_dir, dataset_name+database+".csv")
        dataset = pd.read_csv(dataset_path)["sampleID"].tolist()

        for sampleID in dataset:
            spkrID = sampleID[:-delimiter]
            transcription_path = os.path.join(transcriptions_path, spkrID, sampleID+".txt")
            transcription = open(transcription_path, "r", encoding=enc).readlines()[0].strip().lower()
            clean_transcription = transcription.replace("{", "").replace("}", "")
            if database in ["VLRF", "LIP-RTVE"]:
                clean_transcription = unidecode(clean_transcription.replace("ñ", "N")).replace("N", "ñ")

            for word in clean_transcription.split():
                if word_freq.get(word) is None:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

    return dict(sorted(word_freq.items(), key=lambda elem: elem[1], reverse=True))

def compute_ranks(word_freq):
    word_rank = {}
    top_count = 0
    for idx, (word, frequency) in enumerate(list(word_freq.items())):
            if idx == 0:
                top_count = frequency
            word_rank[word] = frequency/top_count
    return word_rank

def zipf_to_csv(dest_path, vocabulary_ranks, zipf_law, database_rank, database_freq, database_words):
    dest_dir = "/".join(dest_path.split("/")[:-1])
    os.makedirs(dest_dir, exist_ok=True)
    with open(dest_path, "w") as f:
        f.write(f"vocab_id,zipf_law,database_rank,database_freq,database_word\n")
        for vocab, zipf_rank, data_rank, data_freq, data_word in zip(vocabulary_ranks, zipf_law, database_rank, database_freq, database_words):
            f.write(f"{vocab},{zipf_rank},{data_rank},{data_freq},{data_word}\n")

def compare_to_zipf_law_plot(database, word_freq, word_rank, dest_path, i_want_to_plot_it=False):
    vocabulary_ranks = np.arange(len(word_freq)) + 1

    zipf_law = np.array([float(1/r) for r in range(1, len(word_freq)+1)]) # [1/1, 1/2, 1/3, 1/4, ...]
    database_law = word_rank.values() # the actual rank computed based on the database

    zipf_to_csv(dest_path, vocabulary_ranks, zipf_law, database_law, list(word_freq.values()), list(word_freq.keys()))

    fig, ax = plt.subplots()
    ax.loglog(vocabulary_ranks, zipf_law, label="Zipf's law")
    ax.loglog(vocabulary_ranks, database_law, label=database.upper())
    ax.set_xlabel('vocabulary rank')
    ax.set_ylabel('frequency')
    ax.legend()

    fig.savefig(dest_path.replace(".csv", ".png"))

if __name__ == "__main__":
    # -- parsing command line arguments
    parser = argparse.ArgumentParser(description="Computing Zipf's law statistics.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--database", default="LRS2-BBC", type=str, help="Choose a database.")
    parser.add_argument("--scenario", default="speaker-independent", type=str, help="Choose a scenario.")
    parser.add_argument("--datasets", nargs="+", type=str, default=["fulltrain"], help="Data sets to consider")
    parser.add_argument("--text-encoding", default="utf-8", type=str, help="Text encoding type: utf-8 or ISO-8859-1")
    parser.add_argument("--delimiter", default=6, type=int, help="Delimiter to identify the speaker's identifier.")
    parser.add_argument("--dest-path", required=True, type=str, help="Path to save the statistics.")
    args = parser.parse_args()


    # -- computing word frequencies and their corresponding rank w.r.t. the highest-frequency word
    word_freq = compute_zipf_law(args.database, args.scenario, args.datasets, args.text_encoding, args.delimiter)
    word_rank = compute_ranks(word_freq)

    # -- obtaining Zipf's law statistics
    compare_to_zipf_law_plot(args.database, word_freq, word_rank, args.dest_path, i_want_to_plot_it=True)
