# Respiratory Sound Classification

This repository contains the whole process of our experiments.

## 1. Environment installation

`pip install -r requirements.txt`

## 2. Pre-processing

We download SPRSound database from [here](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound). Put the database into ` ./dataset` dirictory, and run the following notebook to divide the database into training set and validation set for each task.

`./dataset_new/data_partitioning.ipynb`

You can put the test set of the two tasks into following file folders respectively:

`./dataset_new/task1_wav/test`

`./dataset_new/task2_wav/test`

## 3. Feature Extraction

We implement Short-Time Fourier Transform(STFT) and Wavelet Analysis to analyse lung sounds, which are executed in `train.py` and `main.py`.

## 4. Train 

The model was built using PyTorch. All details have been emssembled in the `train.sh` and `train.py` files. Please run the command below to train the model:

`sh train.sh`

## 5. Test 

All details have been emssembled in the `test.sh` and `main.py` files. We have saved the trained models for each task in the ` ./results` dirictory, so you can run the command below to test the model derictly:

`sh test.sh`
