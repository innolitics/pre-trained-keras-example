import sys
import json
import argparse
import os
import glob

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import load_model

from train import DataClassifier

num_columns = 6
num_rows = 3

def plot_row_item(image_ax, labels_ax, pixels, top_character_names, top_character_probabilities):
    image_ax.imshow(pixels, interpolation='nearest', aspect='auto')
    y_pos = np.arange(len(top_character_names))*0.11
    labels_ax.barh(y_pos, top_character_probabilities, height=0.1, align='center',
            color='cyan', ecolor='black')
    labels_ax.set_xlim([0,1])
    labels_ax.set_yticks(y_pos)
    labels_ax.set_yticklabels(top_character_names, position=(1,0))
    labels_ax.invert_yaxis()
    labels_ax.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    image_ax.axis('off')

def plot_prediction(pixels, model, data_classifier):
    fig = plt.figure()
    inner = gridspec.GridSpec(2, 1, wspace=0.05, hspace=0, height_ratios=[5, 1.2])
    image_ax = plt.Subplot(fig, inner[0])
    labels_ax = plt.Subplot(fig, inner[1])

    predicted_labels = model.predict(np.array([pixels]), batch_size=1)
    character_name_to_probability = data_classifier.one_hot_decode(predicted_labels[0].astype(np.float64))
    top_character_probability = sorted(character_name_to_probability.items(),
                                       key=lambda item_tup: item_tup[1],
                                       reverse=True)[:3]
    top_character_names, top_character_probabilities = zip(*top_character_probability)
    character_idx = data_classifier.one_hot_index(top_character_names[0])

    plot_row_item(image_ax, labels_ax, pixels, top_character_names, top_character_probabilities)

    fig.add_subplot(image_ax)
    fig.add_subplot(labels_ax)
    return fig


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Visualize predictions for an *.npz file given a model weight file.")
    parser.add_argument('--weight-file', required=True, help="File containing the weights for the model")
    parser.add_argument('--data-directory', required=True, help="Directory containing all input images")
    parser.add_argument('--output-directory', required=True, help="Output directory for generated plots.")
    parser.add_argument('--image-path', required=True, nargs="+", help="*.npz file to generate predictions for. Can be a glob.")
    args = parser.parse_args(sys.argv[1:])
    
    model = load_model(args.weight_file)
    data_classifier = DataClassifier(args.data_directory)

    print("{} input image(s) found. Beginning prediction plotting.".format(len(args.image_path)))

    for image_path in tqdm.tqdm(args.image_path, unit='image'):
        pixels = np.load(image_path)['pixels']
        fig = plot_prediction(pixels, model, data_classifier)
        plt.savefig(os.path.join(args.output_directory, os.path.basename(image_path) + 'predictions.png'))
        plt.close(fig)
