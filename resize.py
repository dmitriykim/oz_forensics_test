import sys
import argparse
import os
from tqdm import tqdm
from PIL import Image


def main(args):
    for root, _, images in tqdm(os.walk(args.input)):
        for img_name in images:
            output_path = os.path.join(args.output, os.path.relpath(root, args.input))
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            img = Image.open(os.path.join(root, img_name)).resize((32, 32))
            img.save(os.path.join(output_path, img_name))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='RAW dataset path.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Resized dataset path.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
