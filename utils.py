import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

import subprocess
import glob
import os

import collections
import json



def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


def clean():
    command = "rm -rf __pycache__"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    _, _ = process.communicate()


def remove_line_images():
    lines = glob.glob("line*.jpg")
    for line in lines:
        os.remove(line)


def save_lines(linedict):
    linedict = collections.OrderedDict(
                sorted(linedict.items())
            )

    for idx, (k, v) in enumerate(linedict.items()):
        cv2.imwrite(f"line{idx}.jpg", v)


def remove_alnum(text):
    result = list([val for val in text if val.isalnum()])
    result = "".join(result)
    return result


def remove_alnum_from_list(list_values):
    for idx in range(len(list_values)):
        list_values[idx] = remove_alnum(list_values[idx])

    return list_values


def save_tesseract_text(text):
    if not os.path.exists("result/"):
        os.mkdir("result/")

    json_string = [text]
    with open('result/tesseract.json', 'w') as handle:
        json.dump(json_string, handle, indent=4)


def save_easyocr_text(text_wi_skew, text_no_skew):
    if not os.path.exists("result/"):
        os.mkdir("result/")

    json_string = {
            "wi_skew": text_wi_skew,
            "no_skew": text_no_skew,
        }

    with open('result/easyocr.json', 'w') as handle:
        json.dump(json_string, handle, indent=4)


def save_kerasocr_text(text_wi_skew, text_no_skew):
    if not os.path.exists("result/"):
        os.mkdir("result/")

    json_string = {
            "wi_skew": text_wi_skew,
            "no_skew": text_no_skew,
        }

    with open('result/kerasocr.json', 'w') as handle:
        json.dump(json_string, handle, indent=4)


def read_result_file(path_type):
    if path_type == "tesseract":
        with open("result/tesseract.json") as handle:
            output = json.loads(handle.read())
        return output

    if path_type == "easyocr":
        with open("result/easyocr.json") as handle:
            output = json.loads(handle.read())
        return output

    if path_type == "kerasocr":
        with open("result/kerasocr.json") as handle:
            output = json.loads(handle.read())
        return output
