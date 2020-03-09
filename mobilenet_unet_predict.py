import keras

from keras.models import model_from_json
from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation.predict import model_from_checkpoint_path, predict, evaluate
import os
import sys
import glob
import cv2

# Remove deprecation warning messages coming from tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():

    # Setup the training parameters, input paths etc.,
    checkpoints_path = "./checkpoints_mobilenet_unet_2class/"
    test_images_path = "./dataset2/images_prepped_test/"
    test_masks_path = "./dataset2/masks_prepped_test_2class/"
    predicted_masks_path = "./dataset2/masks_predicted/"
    predicted_overlay_path = "./dataset2/overlay_predicted/"

    # Enable printing of IoU scores for tests.
    print_test_results = 0

    # Model checkpoint path should exist, else it is an error.
    if not os.path.exists(checkpoints_path):
        print("[ERROR] invalid checkpoints path {}".format(checkpoints_path))
        sys.exit(1)

    # Get Intersection over Union (IoU) results for the test data set.
    # The model is picked up from the latest model in the checkpoints_path.
    # The input images and masks (masks) are picked up from their paths.
    if print_test_results:
        test_results = evaluate(inp_images_dir=test_images_path,
                                masks_dir=test_masks_path,
                                checkpoints_path=checkpoints_path)
        print(test_results)
        print("Test results complete")

    # Create output directory to store predicted masks and overlays.
    # Remove any existing png files.
    if not os.path.exists(predicted_masks_path):
        os.makedirs(predicted_masks_path)

    if not os.path.exists(predicted_overlay_path):
        os.makedirs(predicted_overlay_path)

    for f in glob.glob(predicted_masks_path + '/*.png'):
        os.remove(f)

    for f in glob.glob(predicted_overlay_path + '/*.png'):
        os.remove(f)

    # Predict results for the input images and store in output directory.
    # Store both the predicted masks and the overlaid images with masks.

    # Step 1. Load model from checkpoint path.
    model = model_from_checkpoint_path(checkpoints_path)

    # Step 2. Read images one by one and feed it in into predict function.
    #    Output predicted mask is written to the predicted masks directory.
    #    Assume that input images are in png format.
    for infile in glob.glob(test_images_path + '/*.png'):
        filename = os.path.split(infile)[1]
        outfile = os.path.join(predicted_masks_path, filename)
        overlayfile = os.path.join(predicted_overlay_path, filename)

        print("Predicting mask for file {}, output in {}".format(
            infile, outfile))

        # Create predicted mask from input file, and store it in outfile.
        predict(inp=infile, out_fname=outfile, model=model)

        # Overlay the predicted mask file over input file for display.
        overlay_image_with_mask(image_file=infile,
                                mask_file=outfile,
                                overlay_file=overlayfile)

    print("Prediction complete")


def overlay_image_with_mask(image_file, mask_file, overlay_file):
    """Overlay the mask_file on-top of the image_file and create output_file"""
    # Read input image as background image.
    image = cv2.imread(image_file)

    # Read mask image as overlay image.
    mask = cv2.imread(mask_file)

    # Perform edge detection on the overlay image using CV2 Canny algorithm.
    mask_edge = cv2.Canny(mask, 20, 100)

    # Convert 8-bit image into BGR colorspace.
    mask_edge = cv2.cvtColor(mask_edge, cv2.COLOR_GRAY2BGR)

    # Overlay the edge of the mask onto the original image.
    overlay_image = cv2.addWeighted(image, 1, mask_edge, 1, 0)

    # Write out the overlaid image with mask, return status
    return cv2.imwrite(overlay_file, overlay_image)


if __name__ == "__main__":
    main()
