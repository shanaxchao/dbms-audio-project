# Audio Project
Team Members: Shana Chao, Seohyun Ahn  
Term: Yonsei Fall 2025  
Course: IIE2102 - Industrial Data Management

## Project Purpose
The audio project serves as the capstone project for Yonsei's Industrial Data Management course. The goal is to develop a Python script to take audio input and classify it (ex: language, gender/sex, age, etc.). The main application of this project is using audio data from customers via phone surveys, etc. to recommend products based on their classification. 

## Proposed Technologies
### Language Classification
1. librosa Python library: loading audio data
2. speech-to-text model: specific model to be decided

### Age Classification
tbd

### Sex/Gender Classification
tbd

## Project Extension Proposal
### Training Model Steps
1. Import datasets: sorted by language, age, sex/gender
2. Preprocess data: convert from .wav files to waveforms via librosa, transform audio into Mel-frequency cepstral coefficients (MFCCs)
3. Train neural net/classifier on MFCC features, Random Forest
4. Evaluate accuracy

## Dataset

This project uses the following open-source speech datasets:

### 1. EmoV-DB (SLR115)

* **Description:** Emotional English speech database for synthesis (male and female speakers)
* **Link:** [https://github.com/numediart/EmoV-DB](https://github.com/numediart/EmoV-DB)
* **Citation (BibTeX):**

```
@article{adigwe2018emotional,
  title={The Emotional Voices Database: Towards Controlling the Emotion Dimension in Voice Generation Systems},
  author={Adigwe, Adaeze and Tits, No{\'e} and Haddad, Kevin El and Ostadabbas, Sarah and Dutoit, Thierry},
  journal={arXiv preprint arXiv:1806.09514},
  year={2018}
}
```

### 2. Deeply Korean Read Speech Corpus (SLR97)

* **Description:** Korean read speech recordings with text and vocal sentiment labels, recorded in multiple environments and devices
* **Link:** [https://github.com/deeplyinc/Korean-Read-Speech-Corpus](https://github.com/deeplyinc/Korean-Read-Speech-Corpus)
* **Citation (BibTeX):**

```
@misc{deeply_corpus_kor,
  title={{Deeply Korean Read Speech Corpus}},
  author={Deeply Inc.},
  year={2021},
  url={https://github.com/deeplyinc/Korean-Read-Speech-Corpus}
}
```

> **License Notice:** Please refer to dataset licenses for usage restrictions and attribution requirements.
