'''Builds a model, organizes and loads data, and runs model training.'''
import argparse
from collections import defaultdict
import os
import glob
import random

import keras
import numpy as np

from keras.layers import Input, Average
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

def get_model(pretrained_model, all_character_names):
    if pretrained_model == 'inception':
        model_base = keras.applications.inception_v3.InceptionV3(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
        output = Flatten()(model_base.output)
    elif pretrained_model == 'xception':
        model_base = keras.applications.xception.Xception(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
        output = Flatten()(model_base.output)
    elif pretrained_model == 'resnet50':
        model_base = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
        output = Flatten()(model_base.output)
    elif pretrained_model == 'vgg19':
        model_base = keras.applications.vgg19.VGG19(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
        output = Flatten()(model_base.output)
    elif pretrained_model == 'all':
        input = Input(shape=(*IMG_SIZE, 3))
        inception_model = keras.applications.inception_v3.InceptionV3(include_top=False, input_tensor=input, weights='imagenet')
        xception_model = keras.applications.xception.Xception(include_top=False, input_tensor=input, weights='imagenet')
        resnet_model = keras.applications.resnet50.ResNet50(include_top=False, input_tensor=input, weights='imagenet')

        flattened_outputs = [Flatten()(inception_model.output),
                             Flatten()(xception_model.output),
                             Flatten()(resnet_model.output)]
        output = Concatenate()(flattened_outputs)
        model_base = Model(input, output)

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(all_character_names), activation='softmax')(output)
    model = Model(model_base.input, output)
    for layer in model_base.layers:
        layer.trainable = False
    model.summary(line_length=200)

    # Generate a plot of a model
    import pydot
    pydot.find_graphviz = lambda: True
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='../model_pdfs/{}.pdf'.format(pretrained_model))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

BATCH_SIZE = 64
IMG_SIZE = (256, 256)

image_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=1,
    horizontal_flip=True,
    vertical_flip=False,)

class DataEncoder():
    def __init__(self, all_character_names):
        self.all_character_names = all_character_names

    def one_hot_index(self, character_name):
        return self.all_character_names.index(character_name)

    def one_hot_decode(self, predicted_labels):
        return dict(zip(self.all_character_names, predicted_labels))

    def one_hot_encode(self, character_name):
        one_hot_encoded_vector = np.zeros(len(self.all_character_names))
        idx = self.one_hot_index(character_name)
        one_hot_encoded_vector[idx] = 1
        return one_hot_encoded_vector


class DataGenerator():
    def __init__(self, data_path):
        self.data_path = data_path
        self.partition_to_character_name_to_npz_paths = {
            'train': defaultdict(list),
            'validation': defaultdict(list),
            'test': defaultdict(list),
        }
        self.all_character_names = set()
        npz_file_listing = list(glob.glob(os.path.join(data_path, '**/*.npz')))
        for npz_path in npz_file_listing:
            character_name = os.path.basename(os.path.dirname(npz_path))
            self.all_character_names.add(character_name)
            if hash(npz_path) % 10 < 7:
                partition = 'train'
            elif 7 <= hash(npz_path) % 10 < 9:
                partition = 'validation'
            elif 9 == hash(npz_path) % 10:
                partition = 'test'
            else:
                raise Exception("partition not assigned")
            self.partition_to_character_name_to_npz_paths[partition][character_name].append(npz_path)
        self.encoder = DataEncoder(sorted(list(self.all_character_names)))


    def _pair_generator(self, partition, augmented=True):
        while True:
            for character_name, npz_paths in self.partition_to_character_name_to_npz_paths[partition].items():
                npz_path = random.choice(npz_paths)
                pixels = np.load(npz_path)['pixels']
                one_hot_encoded_labels = self.encoder.one_hot_encode(character_name)
                if augmented:
                    augmented_pixels = next(image_datagen.flow(np.array([pixels])))[0].astype(np.uint8)
                    yield augmented_pixels, one_hot_encoded_labels
                else:
                    yield pixels, one_hot_encoded_labels


    def batch_generator(self, partition, batch_size, augmented=True):
        while True:
            data_gen = self._pair_generator(partition, augmented)
            pixels_batch, one_hot_encoded_character_name_batch = zip(*[next(data_gen) for _ in range(batch_size)])
            pixels_batch = np.array(pixels_batch)
            one_hot_encoded_character_name_batch = np.array(one_hot_encoded_character_name_batch)
            yield pixels_batch, one_hot_encoded_character_name_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', choices={'inception', 'xception', 'resnet50', 'all', 'vgg19'})
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--weight-directory', required=True,
                        help="Directory containing the model weight files")
    parser.add_argument('--tensorboard-directory', required=True,
                        help="Directory containing the Tensorboard log files")
    parser.add_argument('--epochs', required=True, type=int,
                        help="Number of epochs to train over.")
    args = parser.parse_args()

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=args.tensorboard_directory, 
                                                       histogram_freq=0,
                                                       write_graph=True,
                                                       write_images=False)
    save_model_callback = keras.callbacks.ModelCheckpoint(os.path.join(args.weight_directory, 'weights.{epoch:02d}.h5'),
                                                          verbose=3,
                                                          save_best_only=False,
                                                          save_weights_only=False,
                                                          mode='auto',
                                                          period=1)

    data_generator = DataGenerator(args.data_dir)
    model = get_model(args.pretrained_model, data_generator.encoder.all_character_names)

    model.fit_generator(
        data_generator.batch_generator('train', batch_size=BATCH_SIZE),
        steps_per_epoch=200,
        epochs=args.epochs,
        validation_data=data_generator.batch_generator('validation', batch_size=BATCH_SIZE, augmented=False),
        validation_steps=10,
        callbacks=[save_model_callback, tensorboard_callback],
        workers=4,
        pickle_safe=True,
    )
