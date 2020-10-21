# CNNWordReco
## Description
Code and scripts for training and testing an isolated-word speech recogniser using stacked residual CNNs.

## Caveat emptor
This repository is for educational and showcasing purposes only. The contents are provided 'as is' without any implied warranty, and disclaiming liability for damages resulting from using it.

## Functionalities
The code provides 2 functionalities: training a word recogniser and evaluatinf the word recogniser. The depth and size of the convolutional layers can be modified with several arguments. Training can be performed in CPU or GPU, testing is performed on CPU.

### Training
Training is done with the ``train_wordreco.py`` script. Run ``python train_wordreco.py --help`` for details on the existing arguments. An example run is provided here:

```
python3 train_wordreco.py --input_file data/train.csv --output_file output/model.pytorch --xsize 20 --ysize 20 --num_blocks 5 --channels 32 --embedding_size 128 --epochs 15 --batch_size 32 --learning_rate 0.001 --dropout 0.4 --augment --verbose 1
```

The input file is a CSV file containing at least two columns: ``wavfile`` with the path to each wavefile to use, and ``word`` with the word corresponding to each wavefile. The level of output can be modified with the ``--verbose`` argument to 0 (no output), 1 or 2 (full debug output).

### Testing
Testing is done with the ``test_wordreco.py`` scripts. Run ``python test_wordreco.py --help`` for details on the existing arguments. An example run is provided here:

```
python test_wordreco.py --input_file data/test.csv --model_file output/model.pytorch --verbose 1
```

The input CSV file must have at the same columns as in training. The level of output can be modified with the ``--verbose`` argument to 0 (only global accuracy is reported), 1 (full confidence matrix is reported) or 2 (full debug output).

## Dependencies

The code has been tested in the following environment, with the main following dependencies and versions:

* python --> 3.6.7
* torch --> 1.4.0
* numpy --> 1.19.2
* librosa --> 0.7.0
* Pillow --> 8.0.0
* soundfile --> 0.10.3
* scipy --> 1.5.3

## Evaluation

### Data

The code has been evaluated with data from the Free Spoken Digit Dataset (FSDD), available in [Kaggle](https://www.kaggle.com/joserzapata/free-spoken-digit-dataset-fsdd), using one of the avilable speakers for test and training in the remaining speakers.

### Results

This is the confusion matrix obtained with a CNN model on a traing and test from the FSDD. The overall accuracy is 91.20%:

|     |EIGHT| FIVE| FOUR| NINE|  ONE|SEVEN|  SIX|THREE|  TWO| ZERO|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|EIGHT|   46|    0|    0|    0|    0|    0|    2|    2|    0|    0|
| FIVE|    0|   50|    0|    0|    0|    0|    0|    0|    0|    0|
| FOUR|    0|    0|   50|    0|    0|    0|    0|    0|    0|    0|
| NINE|    0|    0|    0|   44|    6|    0|    0|    0|    0|    0|
|  ONE|    0|    0|    0|    0|   50|    0|    0|    0|    0|    0|
|SEVEN|    0|    0|    0|    0|    0|   50|    0|    0|    0|    0|
|  SIX|   13|    0|    0|    0|    0|    0|   20|   16|    0|    1|
|THREE|    0|    0|    0|    0|    0|    0|    0|   49|    1|    0|
|  TWO|    0|    0|    0|    0|    0|    0|    0|    0|   47|    3|
| ZERO|    0|    0|    0|    0|    0|    0|    0|    0|    0|   50|


