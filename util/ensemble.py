import os

import util.const as const

def get_models_ensembled():
    model_names = []

    ensembled_models_path = const.ENSEMBLED_MODELS_PATH

    with open(ensembled_models_path, newline='', ) as f:
        reader = csv.reader(f)
        for row in reader:
            model_names.append(row[0])

    return model_names

def mark_model_ensembled(exp_name):
    ensembled_models_path = const.ENSEMBLED_MODELS_PATH

    # open file in 'append' mode
    with open(ensembled_models_path, 'a', newline='') as f:
        f.write(exp_name)  # insert as the last line

    return

def get_ensemble_weights():
    ensembled_model_names = get_models_ensembled()
    num_models_used = len(ensembled_model_names)

    saved_prob_weight = num_models_used / (num_models_used + 1)
    img_prob_weight   = 1 - saved_prob_weight
    return saved_prob_weight, img_prob_weight
