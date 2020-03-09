# synopsys-2020
Brain MRI segmentation using Mobilenet and UUNet

* requirements.txt are the required python modules
% pip install -r requirements.txt
* Dataset from Kaggle, needs to be preprocessed.
https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
* Some manual preprocessing done to massage masks into a 2 class mask. Some helper scripts were written for that purpose: tiff_to_png.py and png_modify_pixel_value.py
* A bit 1 in the mask represents a tumor pixel, and bit 0 represents non-tumor pixel
* keras_segmentation package from https://github.com/divamgupta/image-segmentation-keras used. Manually selected Mobilenet input pipeline and U-Net for the 2nd stage pipeline

