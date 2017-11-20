'''Using an already trained model, run predictions on unknown images.'''
import json
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from multiprocessing.pool import Pool

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from vis.visualization.saliency import visualize_saliency, visualize_cam
from train import all_character_names, one_hot_index
import random
import scipy.misc

from visualize import sorted_on_performance

output_directory = '/nas/fast2/simpsons_localization'

NUM_ROWS = 5
NUM_COLS = 4


def compute_localization_test(distance_from):
    from keras.models import load_model
    model = load_model('vgg/weights.10.h5')
    # choose 4 random characters
    random.shuffle(all_character_names)
    random_character_names = all_character_names[:4]
    # chose a random image from each one
    random_character_name_to_pixels = {}
    for random_character_name in random_character_names:
        npz_name, _ = sorted_on_performance(random_character_name, distance_from=1)[0]
        random_character_name_to_pixels[random_character_name] = \
            np.load(os.path.join('/nas/fast2/simpsons_dataset_256_256', random_character_name, npz_name))['pixels']

    # stitch the images together
    stiched = np.zeros((512,512, 3))
    items = list(random_character_name_to_pixels.items())
    stiched[0:256, 0:256, :] = items[0][1]
    stiched[256:512, 0:256, :] = items[1][1]
    stiched[0:256, 256:512, :] = items[2][1]
    stiched[256:512, 256:512, :] = items[3][1]
    resized_image_pixels = scipy.misc.imresize(stiched, (256, 256, 3))

    plt.imshow(resized_image_pixels)
    plt.show()
    for random_character_name in random_character_names:
        print(random_character_name)
        cam = visualize_cam(model,
                              layer_idx=-1,
                              filter_indices=[one_hot_index(random_character_name)],
                              seed_input=resized_image_pixels)
        plt.imshow(np.uint8(resized_image_pixels*np.dstack([cam]*3)), aspect='auto')
        plt.show()



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
    plt.savefig('{}/{}_{}.pdf'.format(output_directory, character_name, title))
    # plt.show()
    plt.close('all')

if __name__ == '__main__':
    num_workers = 3
    compute_localization_test(1)