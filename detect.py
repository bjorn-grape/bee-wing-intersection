#!/usr/bin/env python3

#| # Abeille Cool
#| Detection of veins intersections on bee's wings.

#------------------------------------------------

import argparse
import csv
import os
import sys

from IPython import get_ipython

#------------------------------------------------

image = "datas/test/01_inf.jpg"
directory = "results/"

if get_ipython() is None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("-d", "--directory", help="Output directory",
                        type=str, default="results/")
    args = parser.parse_args()
    image = args.image
    directory = args.directory

#------------------------------------------------

if not os.path.exists(directory):
    os.mkdir(directory)

#------------------------------------------------

filename, _ = os.path.splitext(os.path.basename(image))
output_path = os.path.join(directory, f"{filename}.csv")

#------------------------------------------------

with open(output_path, 'w') as csvfile:
    csvfile.write("0,0")
