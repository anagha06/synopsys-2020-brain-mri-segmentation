import keras

from keras.models import model_from_json
from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation.predict import model_from_checkpoint_path, predict, evaluate
import os
import sys

# Remove deprecation warning messages coming from tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():

    # Setup the testing parameters, input paths etc.,
    checkpoints_path = "./checkpoints_mobilenet_unet_2class/"
    test_images_path = "./dataset2/images_prepped_test/"
    test_annotations_path = "./dataset2/annotations_prepped_test_2class/"

    # Model checkpoint path should exist, else it is an error.
    if not os.path.exists(checkpoints_path):
        print("[ERROR] invalid checkpoints path {}".format(checkpoints_path))
        sys.exit(1)

    # Load model from checkpoint path.
    model = model_from_checkpoint_path(checkpoints_path)

    # Get Intersection over Union (IoU) results for the test data set.
    # The model is picked up from the latest model in the checkpoints_path.
    # The input images and masks (annotations) are picked up from their paths.
    test_results = evaluate(inp_images_dir=test_images_path,
                            annotations_dir=test_annotations_path,
                            model=model)

    print(test_results)
    print("Evaluation complete")


if __name__ == "__main__":
    main()
