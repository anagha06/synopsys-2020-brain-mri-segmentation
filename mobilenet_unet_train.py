import keras

from keras.models import model_from_json
from keras_segmentation.models.unet import mobilenet_unet
import os
import fnmatch

# Remove deprecation warning messages coming from tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_num_files_in_directory(dirpath, pattern='*'):
    return len(fnmatch.filter(os.listdir(dirpath), pattern))


def main():
    # Initialize the model. 
    model = mobilenet_unet(n_classes=2, input_height=224, input_width=224)

    # Setup the training parameters, input paths etc.,
    checkpoints_path = "./checkpoints_mobilenet_unet_2class/"
    train_images_path = "./dataset2/images_prepped_train/"
    train_annotations_path = "./dataset2/annotations_prepped_train_2class/"
    val_images_path = "./dataset2/images_prepped_test/"
    val_annotations_path = "./dataset2/annotations_prepped_test_2class/"
    epochs = 50
    batch_size = 8
    # steps_per_epoch should be (number of training images) / (batch_size)
    num_train_images = get_num_files_in_directory(train_images_path, "*.png")
    steps_per_epoch = int(num_train_images / batch_size) + 1

    # Create checkpoints path if it does not exist.
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # Do the training, and store the updated models in the checkpoints_path.
    model.train(
        checkpoints_path=checkpoints_path,
        train_images=train_images_path,
        train_annotations=train_annotations_path,
        val_images=val_images_path,
        val_annotations=val_annotations_path,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        auto_resume_checkpoint=True,
    )

    print("Training complete")


if __name__ == "__main__":
    main()
