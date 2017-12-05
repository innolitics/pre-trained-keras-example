'''Using an already trained model, run predictions on unknown images.'''
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from train import DataClassifier
import pydot
pydot.find_graphviz = lambda: True
from keras.models import load_model

def get_model_predictions(model, data_classifier):

    test_character_name_to_npz_path = {
        **data_classifier.partition_to_character_name_to_npz_paths['test'],
        **data_classifier.partition_to_character_name_to_npz_paths['validation']
    }

    character_to_predictions = {}

    flattened = [(character_name, npz_path) for character_name, npz_paths in test_character_name_to_npz_path.items() for npz_path in npz_paths]
    for character_name, npz_path in tqdm.tqdm(flattened):
        npz_name = os.path.basename(npz_path)
        pixels = np.load(npz_path)['pixels']
        predicted_labels = model.predict(np.array([pixels]), batch_size=1)
        character_name_to_probability = data_classifier.one_hot_decode(predicted_labels[0].astype(np.float64))
        character_to_predictions.setdefault(character_name, {})[npz_name] = character_name_to_probability

    return character_to_predictions

def get_model_predictions_for_npz(model, data_classifier, character_name, npz_name):
    npz_file_path = os.path.join(data_classifier.data_path, character_name, npz_name)
    pixels = np.load(npz_file_path)['pixels']
    predicted_labels = model.predict(np.array([pixels]), batch_size=1)
    return data_classifier.one_hot_decode(predicted_labels[0].astype(np.float64))
