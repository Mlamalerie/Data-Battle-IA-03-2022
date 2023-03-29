import cv2
import pytesseract
import pandas as pd
import numpy as np
from tabula import read_pdf
import matplotlib.pyplot as plt
from image_processing import denoise_image, equalize_histogram, binarize_image
import os
from tools import get_images_paths
from glob import glob
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# =======
# OCR FUNCTIONS
# =======

# get the text from the image using pytesseract
def get_text_from_image_path(image_path: str, denoise=True, binarize=True):

    img = cv2.imread(image_path)
    if denoise:
        img = denoise_image(img)
    if binarize:
        img = binarize_image(img)

    # best config
    special_config = '--psm 12 --oem 1'  # psd 12 = single line of text
    languages_ = 'eng'
    text = pytesseract.image_to_string(img, config=special_config, lang=languages_)
    return text


# =======
# MAIN FUNCTIONS
# =======


# use glob to get all the images in the folder



def get_interesting_images_paths_from_completion_report(images_folder_path: str) -> list:
    images_path = get_images_paths(images_folder_path)
    if len(images_path) <= 1:
        return images_path

    # get text for each image, using multithreading and tqdm progress bar
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = executor.map(


    return 0


def get_interesting_images_paths_from_completion_log(images_folder_path: str) -> list:
    images_path = get_images_paths(images_folder_path)
    if len(images_path) <= 1:
        return images_path


def get_interesting_images_paths(images_folder_path: str, pdf_type="completion_report") -> list:
    """Return images interesting for ocr.

    Args:
        folder_path (str): Path to the folder containing the images. Example: "data/images_to_ocr/15_5-2/316_15_5_2_Completion_report"
        pdf_type (str): Type of pdf. Choose between "completion_log" or "completion_report"
    """
    if pdf_type == "completion_log":
        return get_interesting_images_paths_from_completion_log(images_folder_path)
    elif pdf_type == "completion_report":
        return get_interesting_images_paths_from_completion_report(images_folder_path)
    else:
        raise ValueError("pdf_type must be 'completion_log' or 'completion_report'")


def main_exemple():
    PDF_EXAMPLE_PATH = "data/NO_Quad_15/15_5-2/316_15_5_2_Completion_report.pdf"
    PDF_EXAMPLE_NAME = os.path.basename(PDF_EXAMPLE_PATH).split(".")[0]
    PDF_EXAMPLE_FOLDERNAME = os.path.basename(os.path.dirname(PDF_EXAMPLE_PATH))

    interesting_images_path: list = get_interesting_images_paths(
        r"data\NO_Quad_15_extracted__original_png\15_5-3\207_15_5_3_COMPLETION_REPORT", pdf_type="completion_report")


if __name__ == "__main__":
    main_exemple()
