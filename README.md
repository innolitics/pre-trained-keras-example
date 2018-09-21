# Image Classifier Using Pre-Trained Models in Keras

This repository contains the example code for our [article on pre-trained deep
learning models with Keras][article].

[article]: https://innolitics.com/articles/pretrained-models-with-keras/

Train, predict, visualize, and produce class-activation map animations for deep
learning models in Keras using pre-trained models as their basis.

## Dependencies

- Python 3.5+
- Imagemagick 7+

## Running the Example

### 1. Download the [example dataset][simpsons-kaggle]

[simpsons-kaggle]: https://www.kaggle.com/alexattia/the-simpsons-characters-dataset

### 2. Preprocess the data

```bash
python convert_data.py --data-dir {path-to-data}
```

### 3. Train the model

```bash
python train.py --pretrained_model {model} \
                --data-dir {path-to-data} \
                --weight-directory {path-to-weight-directory} \
                --tensorboard-directory {path-to-tensorboard-logdir} \
                --epochs {max_epochs}
```

### 4. Visualize model predictions

```bash
python visualize.py --weight-file {path-to-weight-file} \
                    --data-directory {path-to-data} \
                    --output-directory {path-to-output-directory} \
                    --image-path {path-to-image-to-visualize}
```

### 5. Generate a CAM plot

```bash
python cam_animation.py --weight-directory {path-to-weight-directory} \
                        --data-directory {path-to-data-directory} \
                        --image-path {path-to-image-to-visualize} \
                        --cam-path {output-path-for-cam-images} \
                        --weight-limit {max-weights-to-plot}

convert -delay 30 -size 256x256 {output-path-for-cam-images}/*.png -loop 0 {final-gif-name}
```

To make the generation of CAM plots easier, you can use the
`./generate_cam_gifs` script. This assumes:

- Data directory is `../data_dir/simpsons_dataset`
- Weight directory is `../data_dir/weights`
- CAM output path is `../data_dir/cam_output/{model}/{character}`
- All names passed into the script are basenames

```bash
# Generate a single CAM plot
./generate_cam_gifs {model} {character} {npz-file}
# Generate CAM plots for the first 100 images of a character
./generate_cam_gifs {model} {character}
```

## About Innolitics

Innolitics is a team of talented software developers with medical and
engineering backgrounds. Our mission is to accelerate progress in medical
imaging by sharing knowledge, creating tools, and providing quality services to
our clients, with the ultimate purpose of improving patient health.  If you are
working on a project that requires image processing or deep learning expertise,
let us know!  We offer [consulting and development services][company-site].

[company-site]: https://innolitics.com/
