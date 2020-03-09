from PIL import Image
import glob
import argparse


def main():
    # Argument handling.
    parser = argparse.ArgumentParser(
        description='Convert tiff to png in a (input) directory')
    parser.add_argument('dirpath',
                        help="Directory which contains tiff files",
                        default="./",
                        type=str)
    args = parser.parse_args()

    # Read in each tif file in directory and convert to PNG file format.
    for name in glob.glob(args.dirpath + '/*.tif'):
        print("Converting file " + name)
        im = Image.open(name)
        name = str(name).rstrip(".tif")
        im.save(name + '.png', 'PNG')

    print("Done conversion")


if __name__ == "__main__":
    main()
