import argparse
import json
import os
import random
from enum import Enum
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Seed for random
seed = 42

# Language dictionary
lang_dict = {
    "en": "English",
    "si": "Sinhala",
    "ta": "Tamil",
    "ur": "Urdu",
    "or": "Odia",
    "hi": "Hindi",
    "as": "Assamese",
    "al": "Albanian",
    "zh": "Chinese",
    "bn": "Bengali",
    "fa": "Persian",
    "ar": "Arabic",
    "ne": "Nepali",
    "dv": "Dhivehi",
}

import os
os.environ["WANDB_DISABLED"] = "True"

class DATASET(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

def read_file(file_name):
    """Read a text file and return a list of lines."""
    with open(file_name, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def dataset_split(input_data, output_data, train_size=0.6, test_size=0.3):
    """Split input and output data into train, validation, and test sets."""
    test_size = test_size / (1 - train_size)

    source = list(filter(None, input_data))
    target = list(filter(None, output_data))

    target_df = pd.DataFrame(target, columns=["Target"])
    source_df = pd.DataFrame(source, columns=["Source"])

    X_train, X_rem, y_train, y_rem = train_test_split(source_df, target_df, train_size=train_size, random_state=1)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=test_size, random_state=1)

    # Convert DataFrames to lists
    X_train_list = X_train["Source"].tolist()
    y_train_list = y_train["Target"].tolist()
    X_valid_list = X_valid["Source"].tolist()
    X_test_list = X_test["Source"].tolist()
    y_valid_list = y_valid["Target"].tolist()
    y_test_list = y_test["Target"].tolist()

    return X_train_list, y_train_list, X_valid_list, y_valid_list, X_test_list, y_test_list

def generate_seeded_dataset(mwp_data, ratio, language):
    """Generate seeded dataset based on the given ratio and language."""
    seed_input = []
    if language == "Chinese":
        for mwp in mwp_data:
            source_line = mwp.strip()
            source_line = jieba.lcut(source_line)
            source_line = "".join(source_line[:round(len(source_line) * ratio)])
            seed_input.append(source_line)
    else:
        for mwp in mwp_data:
            source_line = mwp.strip().split()
            source_line = " ".join(source_line[:round(len(source_line) * ratio)])
            seed_input.append(source_line)
    return seed_input

def create_json_file(data, file_name):
    json_data = {"data": data}
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation for multilingual word predictor")
    parser.add_argument("--src_language", type=str, default="en", help="Source language code")
    parser.add_argument("--tgt_language", type=str, default="zh", help="Target language code")
    parser.add_argument("--mwp_type", type=str, default="wiki", help="Type of multilingual word predictor dataset")
    parser.add_argument("--train_split", type=float, default=0.6, help="Ratio of training data")
    parser.add_argument("--test_split", type=float, default=0.3, help="Ratio of testing data")
    parser.add_argument("--prepared_data_dir", type=str, default="prepared_data", help="Path to the prepared data directory")
    parser.add_argument("--experiment", type=str, default="default", help="Name of the experiment")

    args = parser.parse_args()

    source_language = lang_dict[args.src_language]
    target_language = lang_dict[args.tgt_language]

    task_list = [f"{args.src_language}-{args.tgt_language}", "mwp-gen"]
    task_batch_size = 2

    source_langauge_file = f"dataset/{args.mwp_type}-{source_language}.txt"
    target_langauge_file = f"dataset/{args.mwp_type}-{target_language}.txt"

    output_dataset_path = os.path.join(args.prepared_data_dir, args.experiment)
    Path(output_dataset_path).mkdir(parents=True, exist_ok=True)

    source_data = read_file(source_langauge_file)
    mwp_data = read_file(target_langauge_file)

    # Generate inputs and labels corresponding to the model and dataset
    ratio = 0.25
    seed_input = generate_seeded_dataset(mwp_data, ratio, target_language)

    task_data = [None] * len(task_list)
    task_data[0] = dataset_split(source_data, mwp_data, train_size=args.train_split, test_size=args.test_split)
    task_data[1] = dataset_split(seed_input, mwp_data, train_size=args.train_split, test_size=args.test_split)

    train_set_size = len(task_data[0][0])
    val_set_size = len(task_data[0][2])
    test_set_size = len(task_data[0][4])

    train_combined_data = []
    val_combined_data = []
    test_combined_data = []

    for i in range(0, train_set_size, task_batch_size):
        for task_id in range(len(task_list)):
            for j in range(task_batch_size):
                index = i + j
                if index < train_set_size:
                    train_combined_data.append({
                        "lang1": f"{task_list[task_id]}: {task_data[task_id][DATASET.TRAIN.value * 2][index]}",
                        "lang2": f"{task_list[task_id]}: {task_data[task_id][DATASET.TRAIN.value * 2 + 1][index]}"
                    })


    for i in range(0, val_set_size, task_batch_size):
        for task_id in range(len(task_list)):
            for j in range(task_batch_size):
                index = i + j
                if index < val_set_size:
                    val_combined_data.append({
                        "lang1": f"{task_list[task_id]}: {task_data[task_id][DATASET.VAL.value * 2][index]}",
                        "lang2": f"{task_list[task_id]}: {task_data[task_id][DATASET.VAL.value * 2 + 1][index]}"
                    })

    for i in range(0, test_set_size, task_batch_size):
        for task_id in range(len(task_list)):
            for j in range(task_batch_size):
                index = i + j
                if index < test_set_size:
                    test_combined_data.append({
                        "lang1": f"{task_list[task_id]}: {task_data[task_id][DATASET.TEST.value * 2][index]}",
                        "lang2": f"{task_list[task_id]}: {task_data[task_id][DATASET.TEST.value * 2 + 1][index]}"
                    })


    create_json_file(train_combined_data, "train.json")
    create_json_file(val_combined_data, "validation.json")
    create_json_file(test_combined_data, "test.json")


if __name__ == "__main__":
    main()


# python your_script.py --src_language en --tgt_language si --mwp_type Simple --train_split 0.6 --test_split 0.3 --prepared_data_dir prepared_data --experiment experiment_1
