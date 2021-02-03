# Tensorflow Object Detection API 2.x Template Repo

This repo contains the structure and scripts to *hopefully* put together a Tensorflow Object Detection Project a little easier.

I built a [Sign Language Detection Project](https://github.com/youngsoul/sign-language-detection-with-tfod2) with this template.


## Tensorflow Documentation

You can find great Tensorflow documentation below.

[TF2 Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

## General References

[Towards AI](https://towardsai.net/p/computer-vision/easing-up-the-process-of-tensorflow-2-0-object-detection-api-and-tensorrt)

[Medium](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

## Caveats

* This repo was only tested on MacOS, Catalina

* This repo worked as of, January 23, 2021

* This technology changes so fast, it might not work in the near future.

## Workspace Directory

This directory structure is where most of the image work and specific model training artifacts will be kept.

### exported_models

This directory will hold the model artifacts that are exported after training a model.  Tensorflow provides a script (`exporter_main_v2.py`) to export the checkpoint files into a format that tensorflow can load.  You will be able to find the exported model in: `workspace/exported_models/saved_models/saved_model.pb`

You will also find the saved pipeline configuration file in `workspace/exported_models/pipeline.config`

### images

This directory is meant to contain all of the images for training.  There are a number of sub-directories and the scripts expect certain directories to be available.  These can be changed in the `scripts/model_config.py` but by convention the images sub-directory data structure is assumes.

#### collected-images

This directory contains all of the raw images to train against.

Each subfolder should represent an object detection class.

If you decide to generated augmented images using the `scripts/generate-augmented-images.py` file, it is assumed the files will be stored in the `workspace/images/augmented-images` directory.  They could be placed anywhere.

NOTE: at this point - no pascal-voc.xml file annotations have been made.  The raw images will be broken up into train/test/holdout splits, then annotating the images will be done.

#### augmented-images

This directory contains images originally from the collected-images directory and has an identical directory structure as collected-images but the sub-directories will contain augemented versions of the images.

#### train, test, holdout

After executing the `scripts/train-test-split.py` images will be merged from either the `collected-images` directory or the `augmented-images` directory and merged into the single directory train, test or holdout.

From the train, test, holdout image annotation can be done.  The annotation file is assumed to be in the same directory as the images.

### models

This directory is where the training job will store custom model checkpoints.  The checkpoints are stored in a file driven from the `scripts/model_config.py` file.

```
# model_config.py
CUSTOM_MODEL_DIR_NAME = 'custom_model'
CHECKPOINT_PATH = MODEL_PATH+f'/{CUSTOM_MODEL_DIR_NAME}'
```

### tf-annotations

This directory will contain the training and testing data in the format that is required by Tensorflow.

The script, `scripts/generate_tfrecord.py` is a Tensorflow provided script to convert pascal-vox.xml annotation files to tensorflow TFRecord format.  When completed the `tf-annotations` directory should contain:

* label_map.pbtxt

* test.record

* train.record

### tf-pre-trained-models

This directory contains one, or the many, Tensorflow Object Detection models as a convenience.  If another model is desired, download that model from the Tensorflow Model Zoo and create a similar directory structure.


## Scripts Directory

This directory contains the scripts that will be used to help automate and create workflow around the TFOD Api.

## tf-model-repo

This directory is where the Tensorflow (https://github.com/tensorflow/models.git) repo will be cloned into.





## Steps

### [1] Create a Python 3.8+ Virtual Environment

Create this virtual environment in the root of this project.   

`TensorflowObjDetectionTemplate/venv`

This project was developed with Python 3.8.5

### [2] Activate Virtual Env

ACTIVATE THE VIRTUAL ENV

Or you will pollute your global python environment badly

`pip install -r requirements`

All of the libraries that I used are listed in `setup.py`


### [3] Update scripts/model_config.py file

* update TEMPLATE_ROOT_DIR

to Fully-Qualified Path to this repo root

* Make sure you have a section in the MODELS_CONFIG dictionary for the model you are interested in trying

* Search for TODO and update as appropriate

### [4] Setup Object Detection

Run the script:

`python scripts/setup-tf2-obj-detection.py`

This will perform a number of steps to setup TFOD models.  See the script for details, but the summary list is below:

     1 Clone Tensorflow Repos Dir

     2 Install the Object Detection API

     3 Verify TF ObjDet Install

     4 Fix tf_util bug

     5 install LabelImg

     6 download pretrained weights for selected model

     7 copy base config for model

     8 create label_map.pbtxt file, if it not not already there

     9 update model configuration file

### [5] Collect Images to train on

If you are capturing new images from your webcam, as we did for the sign language tutorial, then I recommend the script below:

For example, to capture 20 images labeled as 'Hello' in the collected-images (from model_config.py) directory you could issue the following command.

`scripts/video_capture_images.py --label Hello --num-images 20`

### [6] Optionally generated augmented images

To generate additional images from the originally collected images, use the `scripts/generate-augmented-images.py` script.  This script will use the images in a folder and generate the specified number of augmented images.

The generate augmented images script is designed to work on images in a single folder and augment each image in that folder with a specified number of images and write the augmented images to an output directory.

Edit the `scripts/generate-augmented-images.py` file to adjust the augmentation parameters.

`python scripts/generate-augmented-images.py --images-dir workspace/images/collected-images/hello --output-dir workspace/images/augmented-images/hello --per-image 4 --prefix hello`

### [7] Create a Train/Test/Holdout split

Create a train/test/holdout split of the collected or augmented images.

The images-dir points to a directory with subfolders and the splitting will happen in each of the subfolders to make sure the train/test/holdout split is evenly distributed across all of the label folders.

You do not need to have a holdout size.

The default test size is 0.2 or 20%.

This script will MOVE files from the images-dir to the output-dir into folders named train/test/holdout

`python scripts/train-test-split.py --images-dir workspace/images/augmented-images --output-dir workspace/images --test-size 0.2 --holdout-size 0.01`

### [8] Label the images

The tool that I have used is called `LabelImg`.  This tool is pip installed during Step 4.

To start this tool, in a terminal type:

```shell
labelImg
```

[1] The first thing I always do is:

`File->Reset All`

This will shutdown the tool, so you will have to restart.

This will reset any previously saved labeling efforts I did previously.

[2] Auto Save Mode

`View->Auto Save Mode`

[3] Open Dir

Open the directory to label.  In this case, since we did the train/test/holdout split prior to labeling you will select each of the train/test directories and label images in each directory.

[4] Change Save Dir

Make sure the save directory is the same as where the images are located.

Then run through each image and label them.

### [9] Create the TFRecords

Tensorflow wants the training and testing data in a tensorflow format.  Tensorflow provides a script to convert images and pascal_voc.xml annotation files to tfrecord format.

The script we are using comes from Tensorflow.

```shell
python scripts/generate_tfrecord.py -x workspace/images/train -l workspace/tf-annotations/label_map.pbtxt -o workspace/tf-annotations/train.record
python scripts/generate_tfrecord.py -x workspace/images/test -l workspace/tf-annotations/label_map.pbtxt -o workspace/tf-annotations/test.record

```

### [10] Train the model

You are now ready to train the model.  Run the script below 

`python scripts/model_train.py`

That script will use the model_config.py values to generate a script to run a Tensorflow script.

E.g.
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=20000

This step will take some time.  As soon as you see it print out the loss for the first 100 steps you know it is in the process of training

### [11] Export the Trained Model

You could either use a checkpoint for the model as suggested by Nicholas or perform a model export as suggested by Tanner.

`python scripts/model_exporter.py`

This will put the exported model into `workspace/exported_models` by default.

### [12] Detect in Video (Thank you Tanner Gilbert)

You are now ready to run inferences.  In the case of the sign language project we want to capture images from the webcam.

NOTE:
#### Increase Detection Box Label Font on Mac

The default font does not exist on my Mac so it loaded_default which was very small.

This is a known issue in Tensorflow Community.  To fix this, I used the answer from [this StackOverflow](https://stackoverflow.com/questions/46950112/how-to-increase-the-font-size-of-the-bounding-box-in-tensorflow-object-detection)

Edit:
`venv/lib/python3.8/site-packages/object_detection/utils/visualization_utils.py`

Change:
`    font = ImageFont.truetype('arial.ttf', 24)`

To:

`    font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 30)`

`python scripts/detect_from_video.py`


### [13] Detect in Image (Thank you Tanner Gilbert)

If you are detecting objects in images use the:

`python script/detect_from_image.py`

See that file for the parameters.

## Congratulations

If you made it this far you should have a working sign language object detection model.

## Additonal Resource Links

### Tensforflow Model Zoo

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md


