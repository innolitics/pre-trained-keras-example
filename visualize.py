import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

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

def sorted_on_performance(character_name, distance_from=0):
    npz_name_to_predictions = character_to_predictions[character_name]
    sorted_predictions = sorted(npz_name_to_predictions.items(), key=lambda items: abs(items[1][character_name]-distance_from))
    return sorted_predictions

def plot_row(npz_name_character_probability_tup, row_index, row_title):
    for idx, (npz_name, character_name_to_probability) in enumerate(npz_name_character_probability_tup):
        top_character_probability = sorted(character_name_to_probability.items(),
                                           key=lambda item_tup: item_tup[1],
                                           reverse=True)[:3]
        top_character_names, top_character_probabilities = zip(*top_character_probability)
        pixels = np.load(os.path.join('simpsons_dataset', character_name, npz_name))['pixels']

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.05, hspace=0, subplot_spec=outer[row_index*num_columns+idx], height_ratios=[5, 1.2])
        image_ax = plt.Subplot(fig, inner[0])
        labels_ax = plt.Subplot(fig, inner[1])
        plot_row_item(image_ax, labels_ax, pixels, top_character_names, top_character_probabilities)
        if idx == 0:
            image_ax.set_title(row_title)

        labels_ax.set_xlabel(npz_name)
        fig.add_subplot(image_ax)
        fig.add_subplot(labels_ax)

if __name__ =='__main__':
    for character_name in tqdm.tqdm(character_to_predictions):
        num_samples = len(character_to_predictions[character_name])
        if num_samples < num_rows*num_columns:
            print("not enough samples for {} to make a useful plot, skipping".format(character_name))
            continue

        accuracy = len(list(filter(lambda v: v[character_name] >= 0.5, character_to_predictions[character_name].values()))) / num_samples
        fig = plt.figure(figsize=(16, 13))
        fig.suptitle('{}. Overall accuracy {:.1f}%. Number Samples {}.'.format(character_name, accuracy*100, num_samples))
        outer = gridspec.GridSpec(num_rows, num_columns, wspace=0.05, hspace=0.3)
        plot_row(sorted_on_performance(character_name, distance_from=1)[:num_columns], 0, "Best performers")
        plot_row(sorted_on_performance(character_name, distance_from=0.5)[:num_columns], 1, "OK performers")
        plot_row(sorted_on_performance(character_name, distance_from=0)[:num_columns], 2,  "Worst performers")
        plt.savefig('results_visualization/{}.pdf'.format(character_name))
        plt.close('all')
