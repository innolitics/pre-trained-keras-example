'''Builds a model, organizes and loads data, and runs model training.'''
from collections import defaultdict


from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

import keras
import numpy as np
import os
import glob
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

def get_model():
    # TODO: make the pretrained model type an argument so we can try different ones.
    pretrained_model = keras.applications.inception_v3.InceptionV3(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
    output = Flatten()(pretrained_model.output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(all_character_names), activation='softmax')(output)
    model = Model(pretrained_model.input, output)
    for layer in pretrained_model.layers:
        layer.trainable = False
    model.summary(line_length=200)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

partition_to_character_name_to_npz_paths = {
    'train': defaultdict(list),
    'validation': defaultdict(list),
    'test': defaultdict(list),
}
all_character_names = set()
npz_file_listing = list(glob.glob('simpsons_dataset/**/*.npz'))
for npz_path in npz_file_listing:
    character_name = os.path.basename(os.path.dirname(npz_path))
    all_character_names.add(character_name)
    if hash(npz_path) % 10 < 7:
        partition = 'train'
    elif 7 <= hash(npz_path) % 10 < 9:
        partition = 'validation'
    elif 9 == hash(npz_path) % 10:
        partition = 'test'
    else:
        raise Exception("partition not assigned")
    partition_to_character_name_to_npz_paths[partition][character_name].append(npz_path)
all_character_names = sorted(list(all_character_names))

def one_hot_encode(character_name):
    one_hot_encoded_vector = np.zeros(len(all_character_names))
    idx = all_character_names.index(character_name)
    one_hot_encoded_vector[idx] = 1
    return one_hot_encoded_vector

def one_hot_decode(predicted_labels):
    return dict(zip(all_character_names, predicted_labels))

if __name__ == '__main__':
    BATCH_SIZE = 8
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

    def data_generator(partition, augmented=True):
        while True:
            for character_name, npz_paths in partition_to_character_name_to_npz_paths[partition].items():
                npz_path = random.choice(npz_paths)
                pixels = np.load(npz_path)['pixels']
                one_hot_encoded_labels = one_hot_encode(character_name)
                if augmented:
                    augmented_pixels = next(image_datagen.flow(np.array([pixels])))[0].astype(np.uint8)
                    yield augmented_pixels, one_hot_encoded_labels
                else:
                    yield pixels, one_hot_encoded_labels


    def batch_generator(partition, batch_size, augmented=True):
        while True:
            data_gen = data_generator(partition, augmented)
            pixels_batch, one_hot_encoded_character_name_batch = zip(*[next(data_gen) for _ in range(batch_size)])
            pixels_batch = np.array(pixels_batch)
            one_hot_encoded_character_name_batch = np.array(one_hot_encoded_character_name_batch)
            yield pixels_batch, one_hot_encoded_character_name_batch


    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensoroard', histogram_freq=0, write_graph=True, write_images=False)
    save_model_callback = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.h5',  verbose=3, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    model = get_model()

    model.fit_generator(
            batch_generator('train', batch_size=BATCH_SIZE),
            # steps_per_epoch=len(npz_file_listing)*0.7 // BATCH_SIZE,
            steps_per_epoch=15,
            epochs=99999,
            validation_data=batch_generator('validation', batch_size=BATCH_SIZE, augmented=False),
            # validation_steps=len(npz_file_listing)*0.2 // BATCH_SIZE
            validation_steps=15,
            callbacks=[save_model_callback, tensorboard_callback]
    )


