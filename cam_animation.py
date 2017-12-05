import sys
import os
import argparse
from glob import glob

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from vis.visualization.saliency import visualize_saliency, visualize_cam

from train import DataClassifier
from predict import get_model_predictions

ROWS = 8
COLS = 7

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
    args = parser.parse_args(sys.argv[1:])

    print('Building data classifier')
    data_classifier = DataClassifier(args.data_directory)
    print("...finished!")

    path_head, npz_name = os.path.split(args.image_path)
    _, character_name = os.path.split(path_head)

    print('Beginning to sort weights in {}'.format(os.path.join(args.weight_directory, '*.h5')))
    weights = sorted(list(glob(os.path.join(args.weight_directory, '*.h5'))))

    fig = plt.figure(figsize=(11, 11))

    print("Beginning CAM plots")
    for idx, weight in enumerate(tqdm.tqdm(weights)):
        model = load_model(weight)
        predictions = get_model_predictions(model, data_classifier, weight+'.predictions.json')
        ax = plt.subplot(ROWS, COLS, idx+1)
        character_idx = data_classifier.one_hot_index(character_name)
        plt.imshow(cam_weighted_image(model, args.image_path, character_idx), aspect='auto')
        ax.axis('off')
        ax.set_title(os.path.basename(weight))
    plt.savefig('{}_{}_cam.pdf'.format(character_name, npz_name))
    plt.close('all')

        # npz_prediction = predictions[character_name][npz_name]

