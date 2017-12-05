import sys
import os
import argparse
from glob import glob

import tqdm
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import load_model
from vis.visualization.saliency import visualize_saliency, visualize_cam

from train import DataClassifier
from predict import get_model_predictions_for_npz
from visualize import plot_row_item


def cam_weighted_image(model, image_path, character_idx):
    pixels = np.load(image_path)['pixels']
    cam = visualize_cam(model, layer_idx=-1,
                        filter_indices=[character_idx],
                        seed_input=pixels)
    return np.uint8(pixels*np.dstack([cam]*3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an animation of class-activation maps")
    parser.add_argument('--weight-directory', required=True,
                        help="Directory containing the model weight files")
    parser.add_argument('--data-directory', required=True,
                        help="Directory containing the input *.npz images")
    parser.add_argument('--image-path', required=True,
                        help="A specific image path to plot CAM for.")
    parser.add_argument('--cam-path', required=True,
                        help="Directory for storing CAM plots.")
    parser.add_argument('--weight-limit', required=False, type=int)
    args = parser.parse_args(sys.argv[1:])

    print('Building data classifier')
    data_classifier = DataClassifier(args.data_directory)
    print("...finished!")

    path_head, npz_name = os.path.split(args.image_path)
    _, character_name = os.path.split(path_head)

    print('Beginning to sort weights in {}'.format(os.path.join(args.weight_directory, '*.h5')))
    weights = sorted(list(glob(os.path.join(args.weight_directory, '*.h5'))))

    print("Beginning CAM plots")
    for idx, weight in enumerate(tqdm.tqdm(weights[:args.weight_limit], unit='weights')):
        if args.weight_limit and idx >= args.weight_limit:
            break
        model = load_model(weight)

        character_idx = data_classifier.one_hot_index(character_name)
        cam = cam_weighted_image(model, args.image_path, character_idx)

        fig = plt.figure()
        inner = gridspec.GridSpec(2, 1, wspace=0.05, hspace=0, height_ratios=[5, 1.2])
        image_ax = plt.Subplot(fig, inner[0])
        labels_ax = plt.Subplot(fig, inner[1])
        character_name_to_probability = get_model_predictions_for_npz(model,
                                                                      data_classifier,
                                                                      character_name,
                                                                      npz_name)
        top_character_probability = sorted(character_name_to_probability.items(),
                                           key=lambda item_tup: item_tup[1],
                                           reverse=True)[:3]
        top_character_names, top_character_probabilities = zip(*top_character_probability)

        plot_row_item(image_ax, labels_ax, cam, top_character_names, top_character_probabilities)
        labels_ax.set_xlabel(npz_name)
        image_ax.set_title(os.path.basename(weight))

        fig.add_subplot(image_ax)
        fig.add_subplot(labels_ax)

        idx_str = str(idx) if idx > 9 else '0'+str(idx)
        plt.savefig(os.path.join(args.cam_path, 'cam_{}.png'.format(idx_str)))
        plt.close(fig)
