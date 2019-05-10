#!/bin/env python3

import argparse
import numpy as np
import pandas as pd
import os
import subprocess
import sys

from tempfile import TemporaryDirectory
from collections import namedtuple
from sklearn.metrics import recall_score, f1_score, accuracy_score


class Point:
    def __init__(self, pt):
        self._pt = tuple(pt)

    def __eq__(self, other):
        if not (self._pt[0] - 30 <= other._pt[0] <= self._pt[0] + 30):
            return False

        return self._pt[1] - 30 <= other._pt[1] <= self._pt[1] + 30

    def __repr__(self):
        return repr(self._pt)


class TestData:
    script = ""

    def __init__(self, image, coords):
        self.name, _ = os.path.splitext(os.path.basename(image))
        self.image = image
        self.expected_coords = coords

    def score(self):
        with TemporaryDirectory(prefix=f"{self.name}-") as result_directory:
            subprocess.run([
                "python",
                TestData.script,
                self.image,
                "--directory", result_directory
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result_path = os.path.join(result_directory, f"{self.name}.csv")
            if not os.path.exists(result_path):
                return (None, None, None)

            score = self._evaluate(result_path)

        return score

    def _evaluate(self, result_path):
        y_true = np.loadtxt(self.expected_coords, delimiter=',')
        y_pred = np.loadtxt(result_path, delimiter=',')

        y_true = np.apply_along_axis(Point, axis=-1, arr=y_true)
        y_pred = np.apply_along_axis(Point, axis=-1, arr=y_pred)

        true_positives = np.size(y_true[y_true == y_pred])
        false_positives = np.size(y_true[y_true != y_pred])

        accuracy = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + np.size(y_true))

        if accuracy == 0 or recall == 0:
            f_score = 0.
        else:
            f_score = 2. * (accuracy * recall) / (accuracy + recall)

        return (
            accuracy,
            recall,
            f_score
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("data_path")
    args = parser.parse_args()

    script = os.path.abspath(args.script)
    data_path = os.path.abspath(args.data_path)

    if not os.path.exists(script):
        print("Script does not exist.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print("Data directory does not exist.")
        sys.exit(1)

    TestData.script = script

    files = os.listdir(data_path)
    images = sorted([
        os.path.join(data_path, image) for image in files if image.endswith(".jpg")
    ])
    csv_files = sorted([
        os.path.join(data_path, csv_file) for csv_file in files if csv_file.endswith(".csv")
    ])

    datas = [TestData(image, csv_file) for image, csv_file in zip(images, csv_files)]
    results = pd.DataFrame(columns=["accuracy", "recall", "f_score"])
    for data in datas:
        results.loc[data.name] = data.score()

    print(results)
