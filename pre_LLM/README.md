# TL Internship Project: SUD
## Multilingual SBD using BERT models

## Table of contents
* [Virtual environment](#virtual-environment)
* [Preprocess data](#preprocess-data)
* [Train and Test](#train-and-test)


## Installation and Usage
### Virtual environment:  

It is **HIGHLY RECOMMENDED** to use a virtualenv or a conda virtual environment for this project:    

1. Setup is fairly easy, open up a terminal and do
    - `python -m venv /path/to/root/dire/.../.../venv_name` or 
    - `conda create --name myenv`

2. Then everytime you want to run the program, just do  
    - `source ./venv/bin/activate` or 
    - `conda activate myenv`

### Preprocess data

Original dataset consists of .TextGrid files. Need to convert and combine them into one .txt file.  
- Install the required libraries, convert the files, and then combine the txt files with combine_txts.sh shell script.  
- Next, we need to remove --EMPTY-- utterances. 
- Last divide the data in to 80:10:10 with balanced distributions for train:valid:test sets.  
- Change path_to_files accordingly.  
- `punc_sym_to_punc.py` removes the stamp of each utterance and convert \<c/> and \<s/> to \, and \. Run if needed.
- Make sure the texts are in order first.

    ```
    $ cd ./sgh_scripts_datasets/
    $ pip install -r ./preprocess/preprocess_dataset/requirements.txt
    $ cd path/to/sgh_directory
    $ python3 TextGrids_to_txts.py
    $ . ./preprocess/preprocess_dataset/combine_txts.sh --path path/to/sgh_txt
    $ python3 ./preprocess/preprocess_dataset/txtremoveEMPTY.py --data-path path/to/directory/with/all.txt
    $ python3 ./train_test_val_split.py --data-path path/to/directory/with/cleaned_all.txt
    $ python3 ./punc_sym_to_punc.py --data-path path/to/directory/with/sgh_train/test/val.txt
    $ python3 ./create_pkl_dataset_new.py --data-path path/to/directory/with/cleaned_sgh_train/test/val.txt
    ```

The dataset folder structure is as follows:  

    ```
    dataset/
        |
        |-sgh/
        |   |
        |   |-xlm-roberta-base/
        |            |-- test_data.pkl
        |            |-- train_data.pkl
        |            |-- valid_data.pkl
        |-...
        | |-...
        |
        |-...
    ```

#### Sample code

    ```
    $ cd ./sgh_scripts_datasets/
    $ pip install -r ./preprocess_dataset_sgh/requirements.txt
    $ cd sgh_dataset/
    $ python3 ../preprocess_dataset_sgh/TextGrids_to_txts.py --data-path ./sgh_TextGrid
    $ . ../preprocess_dataset_sgh/combine_txts.sh --path ./sgh_txt
    $ python3 ../../preprocess_dataset_sgh/txtremoveEMPTY.py --data-path ../
    $ python3 ../../preprocess_dataset_sgh/train_val_split.py --data-path ../
    $ python3 ../../preprocess_dataset_sgh/punc_sym_to_punc.py --data-path ../
    $ python3 ../../preprocess_dataset_sgh/create_pkl_dataset_new.py --data-path ../
    ```

### Train and Test
Run the `main.py` file:

    ```  
    usage: main.py [-h] [--save-model] [--break-train-loop] [--stage STAGE]
                  [--model-path MODEL_PATH] [--data-path DATA_PATH]
                  [--num-epochs NUM_EPOCHS]
                  [--log-level {INFO,DEBUG,WARNING,ERROR}]
                  [--save-n-steps SAVE_N_STEPS] [--force-save]

    arguments for the model

    optional arguments:
      -h, --help            show this help message and exit
      --save-model          save model
      --break-train-loop    prevent training for debugging purposes
      --stage STAGE         load model from checkpoint stage
      --model-path MODEL_PATH
                            path to model directory
      --train-data-path DATA_PATH
                            path to dataset directory containing train_data.pkl
      --val-data-path DATA_PATH
                            path to dataset directory containing valid_data.pkl
      --eval-type EVAL_TYPE
            valid/tst validation dataset or test dataset for evaluation
      --num-epochs NUM_EPOCHS
                            no. of epochs to run the model
      --log-level {INFO,DEBUG,WARNING,ERROR}
                            Logging info to be displayed
      --save-n-steps SAVE_N_STEPS
                            Save after n steps, default=1 epoch
      --force-save          Force save, overriding all settings
      --config CONFIG	Path to config directory
      --action ACTION 	train/val training or testing
          
    ```

#### Sample code
```
## use GPU
$ cd /new_Multilingual-Sentence-Boundary-detection/

## training
$ python3 ./src/main.py --num-epochs 10 --train-data-path /sgh/xlm-roberta-base/ --val-data-path sgh/xlm-roberta-base/ --model-path /punctuator-model/sgh_10epoch- --eval-type valid
or 
$ python3 ./src/main.py --num-epochs 10 --train-data-path /sgh/xlm-roberta-base/ --val-data-path sgh/xlm-roberta-base/ --model-path /punctuator-model/sgh_10epoch- --eval-type tst

## testing
$ python3 ./src/main.py --train-data-path /sgh/xlm-roberta-base/ --val-data-path /sgh/xlm-roberta-base/ --action val --model-path /path/to/model/dir/ --stage sgh_10epoch-xlm-roberta-base-epoch-1.pth --eval-type valid
or
$ python3 ./src/main.py --train-data-path /sgh/xlm-roberta-base/ --val-data-path /sgh/xlm-roberta-base/ --action val --model-path /path/to/model/dir/ --stage sgh_10epoch-xlm-roberta-base-epoch-1.pth --eval-type tst
```
