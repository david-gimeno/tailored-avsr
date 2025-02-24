<h1 align="center"><span style="font-weight:normal">Tailored Design of Audio-Visual<br />Speech Recognition Models using Branchformers</h1>  

  <div align="center">
    
[David Gimeno-Gómez](https://scholar.google.es/citations?user=DVRSla8AAAAJ&hl=en), [Carlos-D. Martínez-Hinarejos](https://scholar.google.es/citations?user=M_EmUoIAAAAJ&hl=en)
</div>

<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Preparation](#preparation) |
[💪 Training](#training) |
[🔮 Inference](#inference) |
[📊 Results](#results) |
[🦒 Model Zoo](#modelzoo) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>


## <a name="tldr"> </a> TL;DR 
<div align="center">
  
  [📜 Arxiv Link](https://www.arxiv.org/abs/2407.06606)
</div>

We proposed a novel framework harnessing the **flexibility** and **interpretability** offered by the Branchformer encoder architecture in the design of **parameter-efficient AVSR systems**. Extensive experiments on English and Spanish AVSR benchmarks covering multiple data conditions and scenarios demonstrated the effectiveness of our proposed method achieving **state-of-the-art recognition rates** while significantly reducing the model complexity.

## <a name="intro"></a> 📘 Introduction

<div align="center"> <img src="docs/tailored_arch.png"> </div>

**Abstract.** _Recent advances in Audio-Visual Speech Recognition have led to unprecedented achievements in the field, improving the robustness of this type of system in adverse, noisy environments. In most cases, this task has been addressed through the design of models composed of two independent encoders, each dedicated to a specific modality. However, while recent works have explored unified audio-visual encoders, determining the optimal cross-modal architecture remains an ongoing challenge. Furthermore, such approaches often rely on models comprising vast amounts of parameters and high computational cost training processes. In this paper, we aim to bridge this research gap by introducing a novel audio-visual framework. Our proposed method constitutes, to the best of our knowledge, the first attempt to harness the flexibility and interpretability offered by encoder architectures, such as the Branchformer, in the design of parameter-efficient AVSR systems. To be more precise, the proposed framework consists of two steps: first, estimating audio- and video-only systems, and then designing a tailored audio-visual unified encoder based on the layer-level branch scores provided by the modality-specific models. Extensive experiments on English and Spanish AVSR benchmarks covering multiple data conditions and scenarios demonstrated the effectiveness of our proposed method. Results reflect how our tailored AVSR system reaches state-of-the-art recognition rates while significantly reducing the model complexity w.r.t. the prevalent approach in the field._


<div align="center"> <img src="docs/branchformer_scores.gif"> </div>

## <a name="preparation"></a> 🛠️ Preparation

- Prepare the **conda environment** to run the experiments:

```
conda create -n tailored-avsr python=3.8
conda activate tailored-avsr
pip install -r requirements.txt
```

- Get access to your **dataset of interest**, preprocessing and saving the data in the following structure:
  
```
LIP-RTVE/
├── WAVs/
│   ├── speaker000/
│   │   ├── speaker000_0000.wav
│   │   ├── speaker000_0001.wav
│   │   ├── ...
│   ├── speaker001/
│   │   ├── ...
│   ├── ...
├── ROIs/
│   ├── speaker000/
│   │   ├── speaker000_0000.npz
│   │   ├── ...
│   ├── ...
├── transcriptions/
│   ├── speaker000/
│   │   ├── speaker000_0000.txt
│   │   ├── ...
│   ├── ...
```

⚠️ **Warning:** Please place the data in the directory `../data/`. If not, you should modify the paths specified in the corresponding CSV split files, e.g., `splits/training/speaker-independent/liprtve.csv`.

## <a name="training"></a> 💪 Training

We will show the most relevant steps necessary to estimate and AVSR system based on our code implementantion. Let consider we want to train a model for the LIP-RTVE dataset. The first, we should do is to estimate our SentencePiece tokenizer as follows:

```
python src/tokenizers/spm/train_spm_model.py \
  --split-path ./splits/training/speaker-independent/liprtve.csv \
  --dst-spm-dir ./src/tokenizers/spm/256vocab/ \
  --spm-name liprtve \
  --vocab-size 256

```

Once we have modified the `` of the configuration file according to our previous step, the following command represents both the training and inference of an Spanish AVSR system on the Spanish LIP-RTVE dataset among other details:

```
python avsr_main.py \
  --training-dataset ./splits/training/speaker-independent/liprtve.csv \
  --validation-dataset ./splits/validation/speaker-independent/liprtve.csv \
  --test-dataset ./splits/test/speaker-independent/liprtve.csv \
  --config-file ./configs/AVSR/conventional_transformer+ctc_spanish.yaml \
  --mode both \
  --output-dir ./exps/avsr/liprtve/ \
  --output-name test-si-liprtve \
  --yaml-overrides training_settins:batch_size:8
```

## <a name="inference"></a> 🔮 Inference

Once we have estimated a model we can perform more inference processes, e.g., incorporating a Language Model during beam search:

```
python avsr_main.py \
  --test-dataset ./splits/test/speaker-independent/liprtve.csv \
  --config-file ./configs/AVSR/conventional_transformer+ctc_spanish.yaml \
  --load-checkpoint ./exps/avsr/liprtve/models/model_average.pth \
  --mode inference \
  --lm-config-file ./configs/LM/lm-spanish.yaml \
  --load-lm ./model_checkpoints/lm/spanish.pth \
  --output-dir ./exps/avsr/liprtve/ \
  --output-name test-si-liprtve+lm \
```

## <a name="modelzoo"></a> 🦒 Model Zoo

The model checkpoints for audio-only, video-only, and audio-visual settings are publicly available in our official Zenodo repository. Please, click [here](https://zenodo.org/records/14645603) to download the checkpoints along with their corresponding tokenizers and configuration files. By following the instructions indicated above for both training and inference, you will be able to evaluate our models and also fine-tune them to your dataset of interest.

## <a name="results"></a> 📊 Results

<div align="center"> <img src="docs/english_results.png"> </div>

Detailed discussion on these results and those achieved for the Spanish VSR benchmark can be found in our [pre-print paper](https://www.arxiv.org/abs/2407.06606)!

## <a name="citation"></a> 📖 Citation

The paper is currently under review for the Elsevier Computer Speech & Language journal. For the moment, if you found useful our work, please cite our preprint paper as follows:

```
@article{gimeno2024tailored,
  author={Gimeno-G{\'o}mez, David and Carlos-D. Martínez-Hinarejos},
  title={{Tailored Design of Audio-Visual Speech Recognition Models using Branchformers}},
  journal={arXiv preprint arXiv:2407.06606},
  volume={},
  pages={}
  year={2024},
  publisher={},
}
```

## <a name="license"></a> 📝 License

This work is protected by [CC BY-NC-ND 4.0 License (Non-Commercial & No Derivatives)](LICENSE)
