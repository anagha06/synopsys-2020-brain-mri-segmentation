import cv2
import glob
import argparse
import os
import sys
import numpy as np


def main():

    # Arguments handling.
    parser = argparse.ArgumentParser(
        description=
        'Change png pixel values in (input) directory, all non-zero values are set to 1'
    )
    parser.add_argument('indir',
                        help="Directory which contains png files",
                        default="./",
                        type=str)
    parser.add_argument('outdir',
                        help="Directory to write output png files",
                        default="./",
                        type=str)
    parser.add_argument('-v',
                        '--value',
                        help="8-bit pixel value to write",
                        default=1,
                        type=int,
                        choices=range(0, 256))

    args = parser.parse_args()

    # Create output directory if it does not exist.
    # If it exists, exit with error.
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        print("[ERROR]  Output directory {} exists! Please re-check.".format(
            args.outdir))
        sys.exit()

    # Do the main conversion by changing nonzero values to only having each
    # pixel in the B position to 1.
    for name in glob.glob(args.indir + '/*.png'):
        filename = os.path.split(name)[1]
        outfile = os.path.join(args.outdir, filename)
        print("Converting input {} to output {}".format(name, outfile))
        # Input image read into an array.
        img = cv2.imread(name)

        # Output image initialized with zeros.
        img1 = np.zeros(img.shape)

        # For each value which is nonzero, set only the B component of the RGB value to 1.
        for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                if img[y, x, 0] > 0:
                    img1[y, x, 0] = args.value

        # Write to output file.
        cv2.imwrite(outfile, img1)

    print("Done conversion")


if __name__ == "__main__":
    main()
