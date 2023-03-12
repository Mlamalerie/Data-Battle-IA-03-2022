from pdf2image import convert_from_path
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time


def generate_images_from_pdf(resolution: int, pdf_path: str, output_folder_path: str) -> int:
    """Generate images from a pdf

    Args:
        resolution (int): Resolution of the images
        pdf_path (str): Path to the pdf
        output_folder_path (str): Path to the folder where the images will be saved

    Returns:
        int: Number of images generated
    """
    # extract images from pdf
    images = convert_from_path(
        pdf_path, size=resolution, poppler_path=r"poppler-22.04.0/Library/bin"
    )

    pdf_parent_dirname = pdf_path.split("\\")[-2]
    pdf_name = pdf_path.split("\\")[-1].split(".")[0]
    # create folder for the pdf
    os.makedirs(f"{output_folder_path}/{pdf_parent_dirname}/{pdf_name}", exist_ok=True)

    # save images use multithreading (map) with tqdm progress bar
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = executor.map(
            lambda i: images[i].save(
                f"{output_folder_path}/{pdf_parent_dirname}/{pdf_name}/{pdf_name}_{i}.jpg"
            ),
            range(len(images))
        )
    return len(images)


def generate_images(resolution: int, input_folder_path: str) -> None:
    """Generate images from pdfs in a folder

    Args:
        resolution (int): Resolution of the images
        input_folder_path (str): Path to the folder containing the pdfs. Example: "data/NO_Quad_15"
    """
    print(f"* Input folder: {input_folder_path}\n* Resolution: {resolution}")
    # get all pdfs
    pdfs = glob.glob(f"{input_folder_path}/**/*.pdf", recursive=True)
    input_dirname = input_folder_path.split("\\")[-1]
    # create output folder with the same name as the input folder + parameters
    output_dirname = f"{input_dirname}_extracted__{resolution}"
    output_folder_path = input_folder_path.replace(input_dirname, output_dirname)
    os.makedirs(output_folder_path, exist_ok=True)

    count_images = 0
    # generate images from pdfs
    for pdf_path in tqdm(pdfs):
        count_images += generate_images_from_pdf(resolution, pdf_path, output_folder_path)
    # total images generated
    print(f"* Output folder: {output_folder_path}")
    print(f"* Total images generated: {count_images}, Total pdfs: {len(pdfs)}")


if __name__ == "__main__":
    generate_images(3000, "data/NO_Quad_15")
