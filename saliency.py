'''Calculate class activation maps for a set of images and their top 3 possible classes.'''
import json
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from multiprocessing.pool import Pool

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from vis.visualization.saliency import visualize_saliency, visualize_cam
from train import one_hot_index, all_character_names
from visualize import sorted_on_performance

output_directory = '/nas/fast2/simpsons_saliency'

NUM_ROWS = 5
NUM_COLS = 4


def compute_saliencies(distance_from, title):
    from keras.models import load_model
    model = load_model('vgg/weights.10.h5')
    for character_name in tqdm.tqdm(all_character_names):
        npz_name_character_probability_tup = sorted_on_performance(character_name, distance_from=distance_from)
        compute_saliency(model, character_name, npz_name_character_probability_tup, title=title)


def compute_saliency(model, character_name, npz_name_character_probability_tup, title):
    fig = plt.figure(figsize=(11,11))
    for row_idx, (npz_name, character_name_to_probability) in enumerate(npz_name_character_probability_tup[:NUM_ROWS]):
        top_character_probability = sorted(character_name_to_probability.items(),
                                           key=lambda item_tup: item_tup[1],
                                           reverse=True)[:NUM_COLS-1]
        npz_path = os.path.join('/nas/fast2/simpsons_dataset_256_256', character_name, npz_name)
        pixels = np.load(npz_path)['pixels']

        start_idx = row_idx*NUM_COLS
        ax = plt.subplot(NUM_ROWS, NUM_COLS, start_idx+1)
        plt.imshow(pixels, aspect='auto')
        ax.axis('off')
        ax.set_title("Original {}".format(npz_name))
        for idx, (predicted_character_name, probability) in enumerate(top_character_probability):
            cam = visualize_cam(model,
                                  layer_idx=-1,
                                  filter_indices=[one_hot_index(predicted_character_name)],
                                  seed_input=pixels)
            ax = plt.subplot(NUM_ROWS, NUM_COLS, start_idx+idx+2)
            plt.imshow(np.uint8(pixels*np.dstack([cam]*3)), aspect='auto')
            ax.axis('off')
            ax.set_title("{} ({:.1f})".format(predicted_character_name, probability))
    fig.tight_layout()
    plt.savefig('{}/{}/{}.pdf'.format(output_directory, title, character_name))
    # plt.show()
    plt.close('all')

if __name__ == '__main__':
    num_workers = 3
    pool = ProcessPoolExecutor(max_workers=3)
    pool.map(compute_saliencies, (1, 0.5, 0), ('Good', 'Bad', 'Ugly'))
