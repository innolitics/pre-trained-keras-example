"""
Reads image files, decodes jpeg, resizes them, and stores them into an npz so
these operations do not have to be performed more than once.
"""
import numpy as np
# import keras
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.misc
import os
import glob
import tqdm

from collections import defaultdict

IMG_SIZE = (256, 256)

character_name_to_pixels = defaultdict(list)

for image_path in tqdm.tqdm(list(glob.glob('simpsons_dataset/**/*.jpg'))):
    image_pixels = scipy.ndimage.imread(image_path)
    resized_image_pixels = scipy.misc.imresize(image_pixels, IMG_SIZE)
    image_basepath, _ = os.path.splitext(image_path)
    np.savez(image_basepath+'.npz', pixels=resized_image_pixels, compressed=True)
