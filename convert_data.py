"""
Reads image files, decodes jpeg, resizes them, and stores them into an npz so
these operations do not have to be performed more than once.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.ndimage
import scipy.misc
import tqdm

IMG_SIZE = (256, 256)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess data for Simpsons classifier")
    parser.add_argument('--data-dir', required=True, help="Directory of input data")
    args = parser.parse_args(sys.argv[1:])

    character_name_to_pixels = defaultdict(list)

    input_data = list(glob.glob(os.path.join(args.data_dir, '**/*.jpg')))

    for image_path in tqdm.tqdm(input_data):
        image_pixels = scipy.ndimage.imread(image_path)
        resized_image_pixels = scipy.misc.imresize(image_pixels, IMG_SIZE)
        image_basepath, _ = os.path.splitext(image_path)
        np.savez(image_basepath+'.npz', pixels=resized_image_pixels, compressed=True)
