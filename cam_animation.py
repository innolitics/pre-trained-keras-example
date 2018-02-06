import sys
import os
import argparse

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import load_model
from vis.visualization.saliency import visualize_cam

from train import DataGenerator
from visualize import plot_row_item

def get_model_predictions_for_npz(model, data_generator, character_name, npz_name):
    npz_file_path = os.path.join(data_generator.data_path, character_name, npz_name)
    pixels = np.load(npz_file_path)['pixels']
    predicted_labels = model.predict(np.array([pixels]), batch_size=1)
    return data_generator.encoder.one_hot_decode(predicted_labels[0].astype(np.float64))

def cam_weighted_image(model, image_path, character_idx):
    pixels = np.load(image_path)['pixels']
    cam = visualize_cam(model, layer_idx=-1,
                        filter_indices=[character_idx],
                        seed_input=pixels)
    return np.uint8(pixels*np.dstack([cam]*3))

def make_cam_plot(model, weight, image_path, cam_path, data_generator):
    path_head, npz_name = os.path.split(image_path)
    _, character_name = os.path.split(path_head)

    model_name = os.path.basename(os.path.dirname(weight))

    character_idx = data_generator.encoder.one_hot_index(character_name)
    cam = cam_weighted_image(model, image_path, character_idx)

    fig = plt.figure()
    inner = gridspec.GridSpec(2, 1, wspace=0.05, hspace=0, height_ratios=[5, 1.2])
    image_ax = plt.Subplot(fig, inner[0])
    labels_ax = plt.Subplot(fig, inner[1])
    character_name_to_probability = get_model_predictions_for_npz(model,
                                                                  data_generator,
                                                                  character_name,
                                                                  npz_name)
    top_character_probability = sorted(character_name_to_probability.items(),
                                       key=lambda item_tup: item_tup[1],
                                       reverse=True)[:3]
    top_character_names, top_character_probabilities = zip(*top_character_probability)

    plot_row_item(image_ax, labels_ax, cam, top_character_names, top_character_probabilities)
    weight_idx = os.path.basename(weight).split('.')[1]
    labels_ax.set_xlabel(npz_name)
    image_ax.set_title(model_name + ', epoch ' + weight_idx)

    fig.add_subplot(image_ax)
    fig.add_subplot(labels_ax)

    plt.savefig(os.path.join(cam_path, 'cam_{}.png'.format(weight_idx)))
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an animation of class-activation maps")
    parser.add_argument('--weight-file', required=True,
                        help="Model weight file")
    parser.add_argument('--data-directory', required=True,
                        help="Directory containing the input *.npz images")
    parser.add_argument('--cam-path', required=True,
                        help="Directory for storing CAM plots.")
    parser.add_argument('--images', required=True, nargs="+",
                        help="Images to plot CAM for.")
    args = parser.parse_args(sys.argv[1:])

    data_generator = DataGenerator(args.data_directory)


    model = load_model(args.weight_file)
    for image in tqdm.tqdm(args.images, unit="image"):
        try:
            image_cam_path = os.path.join(args.cam_path, os.path.basename(image))
            os.makedirs(image_cam_path)
        except OSError as err:
            if err.errno != os.errno.EEXIST:
                raise err

        make_cam_plot(model, args.weight_file, image, image_cam_path, data_generator)
