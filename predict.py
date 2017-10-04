import random
import numpy as np
import pydot
import matplotlib.pyplot as plt

from train import partition_to_character_name_to_npz_paths, one_hot_decode

pydot.find_graphviz = lambda: True
from keras.models import load_model

model = load_model('weights.505.h5')

# from keras.utils import plot_model
# plot_model(model, to_file='model.pdf')

test_character_name_to_npz_path = partition_to_character_name_to_npz_paths['test']

for character_name, npz_paths in test_character_name_to_npz_path.items():
    random_npz_path = random.choice(npz_paths)
    pixels = np.load(random_npz_path)['pixels']
    predicted_labels = model.predict(np.array([pixels]), batch_size=1)
    character_name_to_probability = one_hot_decode(predicted_labels[0])
    top_character_probability = sorted(character_name_to_probability.items(),
                                       key=lambda item_tup: item_tup[1],
                                       reverse=True)[:3]
    top_character_names, top_character_probabilities = zip(*top_character_probability)
    print(top_character_probability)

    ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=6)
    ax1.set_title(character_name)
    ax2 = plt.subplot2grid((7, 1), (6, 0), rowspan=1)
    ax1.imshow(pixels)
    y_pos = np.arange(len(top_character_names))*0.11
    ax2.barh(y_pos, top_character_probabilities, height=0.1, align='center',
            color='cyan', ecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_character_names, position=(1,0))
    ax2.invert_yaxis()  # labels read top-to-bottom
    plt.show()
