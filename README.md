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

## <a name="intro"></a> 📘 Introduction

Official source code for the paper. Detailed documentations will be available as soon as possible!

<div align="center"> <img src="docs/tailored_arch.png"> </div>

**Abstract.** _Recent advances in Audio-Visual Speech Recognition (AVSR) have led to unprecedented achievements in the field, improving the robustness of this type of system in adverse, noisy environments. In most cases, this task has been addressed through the design of models composed of two independent encoders, each dedicated to a specific modality. However, while recent works have explored unified audio-visual encoders, determining the optimal cross-modal architecture remains an ongoing challenge. Furthermore, such approaches often rely on models comprising vast amounts of parameters and high computational cost training processes. In this paper, we aim to bridge this research gap by introducing a novel audio-visual framework. Our proposed method constitutes, to the best of our knowledge, the first attempt to harness the flexibility and interpretability offered by encoder architectures, such as the Branchformer, in the design of parameter-efficient AVSR systems. To be more precise, the proposed framework consists of two steps: first, estimating audio- and video-only systems, and then designing a tailored audio-visual unified encoder based on the layer-level branch scores provided by the modality-specific models. Extensive experiments on English and Spanish AVSR benchmarks covering multiple data conditions and scenarios demonstrated the effectiveness of our proposed method. Results reflect how our tailored AVSR system is able to reach state-of-the-art recognition rates while significantly reducing the model complexity w.r.t. the prevalent approach in the field._


## <a name="preparation"></a> 🛠️ Preparation

## <a name="training"></a> 💪 Training

## <a name="inference"></a> 🔮 Inference

## <a name="results"></a> 📊 Results

## <a name="modelzoo"></a> 🦒 Model Zoo

The model checkpoints for audio-only, video-only, and audio-visual settings are publicly available in our official Zenodo repository. Please, click [here](https://zenodo.org/records/11441180]) to download the checkpoints along with their corresponding configuration files. By following the instructions indicated above for both training and inference, you will be able to evaluate our models and also fine-tune them to your database of interest.

## <a name="citation"></a> 📖 Citation

## <a name="license"></a> 📝 License
