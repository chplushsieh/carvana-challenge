import os
import csv

import numpy as np

import util.const as const

def create_file_if_not_exist(file_path):
    # create empty file if it doesn't exist
    if not os.path.isfile(file_path):
        open(file_path, 'a').close()

    return


def get_models_ensembled(ensemble_dir):
    model_names = []
    test_time_aug_names = []

    ensembled_models_path = os.path.join(const.OUTPUT_DIR, ensemble_dir, 'models_ensembled.txt')

    create_file_if_not_exist(ensembled_models_path)

    # read file content
    with open(ensembled_models_path, newline='', ) as f:
        reader = csv.reader(f)
        for row in reader:
            model_names.append(row[0])
            test_time_aug_names.append(row[1])

    return model_names, test_time_aug_names

def mark_model_ensembled(ensemble_dir, exp_name, test_time_aug_name):
    ensembled_models_path = os.path.join(const.OUTPUT_DIR, ensemble_dir, 'models_ensembled.txt')

    create_file_if_not_exist(ensembled_models_path)

    # open file in 'append' mode
    with open(ensembled_models_path, 'a', newline='') as f:
        f.write(exp_name + ',' + test_time_aug_name + '\n')  # insert as the last line

    return

def get_ensemble_weights(ensemble_dirs):
    total_models = 0
    weights = np.zeros(len(ensemble_dirs))

    for i, ensemble_dir in enumerate(ensemble_dirs):
        ensembled_model_names, _ = get_models_ensembled(ensemble_dir)
        num_models_used = len(ensembled_model_names)

        total_models += num_models_used
        weights[i] = num_models_used

    weights = np.divide(weights, total_models)
    return weights
