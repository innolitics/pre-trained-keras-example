'''Using an already trained model, run predictions on unknown images.'''
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from train import partition_to_character_name_to_npz_paths, one_hot_decode
import pydot
pydot.find_graphviz = lambda: True
from keras.models import load_model

model = load_model('vgg/weights.10.h5')

# from keras.utils import plot_model
# plot_model(model, to_file='model.pdf')

test_character_name_to_npz_path = {**partition_to_character_name_to_npz_paths['test'], **partition_to_character_name_to_npz_paths['validation']}

output_json_name = 'all_predictions.json'
character_to_predictions = {}
json.dump(character_to_predictions, open(output_json_name, 'w'))

flattened = [(character_name, npz_path) for character_name, npz_paths in test_character_name_to_npz_path.items() for npz_path in npz_paths]
for character_name, npz_path in tqdm.tqdm(flattened):
    npz_name = os.path.basename(npz_path)
    pixels = np.load(npz_path)['pixels']
    predicted_labels = model.predict(np.array([pixels]), batch_size=1)
    character_name_to_probability = one_hot_decode(predicted_labels[0].astype(np.float64))
    character_to_predictions.setdefault(character_name, {})[npz_name] = character_name_to_probability

json.dump(character_to_predictions, open(output_json_name, 'w'))
