# Audio Project
Team Members: Shana Chao, Seohyun Ahn  
Term: Yonsei Fall 2025  
Course: IIE2102 - Industrial Data Management

## Project Purpose
The audio project serves as the capstone project for Yonsei's Industrial Data Management course. The goal is to develop a Python script to take audio input and classify it (ex: language, gender/sex, age, etc.). The main application of this project is using audio data from customers via phone surveys, etc. to recommend products based on their classification. 

## Proposed Technologies
### Language Classification
1. torchaudio Python library: loading audio data
2. speech-to-text model: specific model to be decided

### Age Classification
tbd

### Sex/Gender Classification
tbd

## Project Extension Proposal
### Training Model Steps
1. Import datasets: sorted by language, age, sex/gender
2. Preprocess data: convert from MP3 to waveforms via torchaudio/librosa, transform audio into Mel-frequency cepstral coefficients (MFCCs)
3. Train neural net/classifier on MFCC features
4. Evaluate accuracy

