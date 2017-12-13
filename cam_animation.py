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

# Debug purposes only
# from pympler import muppy, summary 

def cam_weighted_image(model, image_path, character_idx):
    pixels = np.load(image_path)['pixels']
    cam = visualize_cam(model, layer_idx=-1,
                        filter_indices=[character_idx],
                        seed_input=pixels)
    return np.uint8(pixels*np.dstack([cam]*3))

def make_cam_plot(model, weight, image_path, cam_path, data_classifier):
    path_head, npz_name = os.path.split(image_path)
    _, character_name = os.path.split(path_head)

    # print("generating CAM image")
    character_idx = data_classifier.one_hot_index(character_name)
    cam = cam_weighted_image(model, image_path, character_idx)

    fig = plt.figure()
    inner = gridspec.GridSpec(2, 1, wspace=0.05, hspace=0, height_ratios=[5, 1.2])
    image_ax = plt.Subplot(fig, inner[0])
    labels_ax = plt.Subplot(fig, inner[1])
    # print("Getting model predictions")
    character_name_to_probability = get_model_predictions_for_npz(model,
                                                                  data_classifier,
                                                                  character_name,
                                                                  npz_name)
    top_character_probability = sorted(character_name_to_probability.items(),
                                       key=lambda item_tup: item_tup[1],
                                       reverse=True)[:3]
    top_character_names, top_character_probabilities = zip(*top_character_probability)

    # print("Plotting CAM image")
    plot_row_item(image_ax, labels_ax, cam, top_character_names, top_character_probabilities)
    weight_idx = os.path.basename(weight).split('.')[1]
    labels_ax.set_xlabel(npz_name)
    image_ax.set_title(model_name + ', epoch ' + weight_idx)

    fig.add_subplot(image_ax)
    fig.add_subplot(labels_ax)

    # print("Saving CAM image")
    plt.savefig(os.path.join(cam_path, 'cam_{}.png'.format(weight_idx)))
    plt.close(fig)
    # print("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an animation of class-activation maps")
    # parser.add_argument('--weight-directory', required=True,
    #                     help="Directory containing the model weight files")
    parser.add_argument('--weight-file', required=True,
                        help="Model weight file")
    parser.add_argument('--data-directory', required=True,
                        help="Directory containing the input *.npz images")
    parser.add_argument('--image-path', required=True,
                        help="A specific image path to plot CAM for.")
    parser.add_argument('--cam-path', required=True,
                        help="Directory for storing CAM plots.")
    # parser.add_argument('--weight-limit', required=False, type=int)
    args = parser.parse_args(sys.argv[1:])

    # print('Building data classifier')
    data_classifier = DataClassifier(args.data_directory)
    # print("...finished!")

    model_name = os.path.basename(os.path.dirname(args.weight_file))

    # print("loading model")
    model = load_model(args.weight_file)
    make_cam_plot(model, args.weight_file, args.image_path, args.cam_path, data_classifier)

    # path_head, npz_name = os.path.split(args.image_path)
    # _, character_name = os.path.split(path_head)

    # print('Beginning to sort weights in {}'.format(os.path.join(args.weight_directory, '*.h5')))
    # weights = sorted(list(glob(os.path.join(args.weight_directory, '*.h5'))))



    # print("Beginning CAM plots")

    # start = summary.summarize(muppy.get_objects())

    # for idx, weight in enumerate(tqdm.tqdm(weights[:args.weight_limit], unit='weights')):
    #     # cursor = summary.summarize(muppy.get_objects())
    #     # summary.print_(summary.get_diff(start, cursor))
    #     if args.weight_limit and idx >= args.weight_limit:
    #         break

    #     del top_character_names, top_character_probabilities, top_character_probability, character_name_to_probability
    #     del cam
    #     del fig
    #     del model
