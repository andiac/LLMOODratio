# LLMOODratio

This repository contains the code and resources for the paper "Your Finetuned Large Language Model is Already a Powerful Out-of-distribution Detector" [(arxiv)](https://arxiv.org/abs/2404.08679).

## Overview

Our work demonstrates that finetuned Large Language Models (LLMs) can serve as effective Out-of-Distribution (OOD) detectors without additional modifications or training. We provide evidence through various experiments that LLMs inherently possess the capability to identify inputs that deviate from their training distribution.

## Repository Structure

The repository is organized into three main directories, each corresponding to a specific experiment from our paper:

- **OOD/:** Contains code and datasets for both far-OOD and near-OOD experiments (Sections 5.1 and 5.2)
- **SPAM/:** Resources for the spam detection experiment (Section 5.3)
- **QA/:** Implementation of OOD question detection in QA systems (Section 5.4)

## Experiments

### Near and far OOD detection

To reproduce our Near and Far OOD detection experiments (Sections 5.1 & 5.2), follow these steps:

1. Navigate to the OOD experiments directory:

```
cd OOD
```

2. Run the training script to finetune the pre-trained LLM on in-distribution data:

```
bash train.sh
```

This will finetune the model and calculate sequence likelihoods, storing the results in `.pkl` files.

3. Evaluate the OOD detection performance:

```
bash eval.sh
```

This script processes the likelihood files and outputs comprehensive OOD detection metrics including AUROC, AUPR, and FPR95.

### Spam Detection
To replicate our spam detection experiments (Section 5.3), follow these steps:

1. Navigate to the spam detection directory:

```
cd SPAM
```

2. Run the comprehensive experiment script:

```
bash go.sh
```

### OOD Question Detection in QA Systems

To reproduce our experiments on detecting out-of-distribution questions in QA systems (Section 5.4), follow these steps:

1. Navigate to the QA experiments directory:

```
cd QA
```

2. Execute the experiment script:

```
bash go.sh
```
