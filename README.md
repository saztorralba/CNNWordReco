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
python3 train_wordreco.py --input_file data/train.csv --output_file output/model.pytorch --xsize 20 --ysize 20 --num_blocks 10 --channels 32 --embedding_size 128 --epochs 20 --batch_size 32 --learning_rate 0.001 --dropout 0.3 --verbose 1
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

The code has been evaluated with data from the Free Spoken Digit Dataset (FSDD), available in [GitHub](https://github.com/Jakobovski/free-spoken-digit-dataset/),
and with a task available in [Kaggle](https://www.kaggle.com/joserzapata/free-spoken-digit-dataset-fsdd).
The original author proposes to use 2,700 recordings for training (45 per speaker per digit) and 300 recordings for testing (5 per speaker per digit).

### Results

On the proposed setup, the accuracy obtained is 97.67%, and the confusion matrix achieved is:

|     | ZERO|  ONE|  TWO|THREE| FOUR| FIVE|  SIX|SEVEN|EIGHT| NINE|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| ZERO|   30|    0|    0|    0|    0|    0|    0|    0|    0|    0|
|  ONE|    0|   30|    0|    0|    0|    0|    0|    0|    0|    0|
|  TWO|    0|    0|   29|    1|    0|    0|    0|    0|    0|    0|
|THREE|    0|    0|    0|   28|    0|    0|    2|    0|    0|    0|
| FOUR|    0|    0|    0|    0|   30|    0|    0|    0|    0|    0|
| FIVE|    0|    0|    0|    0|    0|   29|    1|    0|    0|    0|
|  SIX|    0|    0|    0|    0|    0|    0|   30|    0|    0|    0|
|SEVEN|    0|    0|    0|    0|    0|    0|    0|   30|    0|    0|
|EIGHT|    0|    0|    0|    0|    0|    0|    2|    0|   28|    0|
| NINE|    0|    1|    0|    0|    0|    0|    0|    0|    0|   29|


