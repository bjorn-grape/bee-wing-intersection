#!/usr/bin/env python3

import argparse
import csv

from PIL import Image, ImageDraw
from numpy import loadtxt, array


def draw_intersections(image, intersections):
    draw = ImageDraw.Draw(image)
    for x, y in intersections:
        draw.ellipse((y - 10, x - 10, y + 10, x + 10), fill=(255, 0, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("csv")
    parser.add_argument("output")
    args = parser.parse_args()

    image_path = args.image
    csv_path = args.csv
    output_path = args.output

    image = Image.open(image_path)
    coords = loadtxt(csv_path, delimiter=',')
    if len(coords.shape) != 2:
        coords = array([coords])

    draw_intersections(image, coords)

    image.save(output_path)
