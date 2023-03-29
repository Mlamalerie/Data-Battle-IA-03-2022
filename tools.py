from pdf2image import convert_from_path
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import PIL
import time
PIL.Image.MAX_IMAGE_PIXELS = None

def get_images_paths(folder_path: str, extension="png"):
    images_path = glob.glob(f"{folder_path}/*.{extension}")
    images_path = sorted(images_path, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return images_path

def generate_images_from_pdf(pdf_path: str, output_folder_path: str, extension: str = "png", resolution: int = None, overwrite : bool = False ) -> int:
    """Generate images from a pdf

    Args:
        resolution (int): Resolution of the images
        pdf_path (str): Path to the pdf
        output_folder_path (str): Path to the folder where the images will be saved

    Returns:
        int: Number of images generated
    """
    # pdf exists
    if not os.path.exists(pdf_path):
        raise ValueError(f"File {pdf_path} does not exist")

    pdf_path = pdf_path.replace("/", "\\") # replace / with \ for windows
    pdf_parent_dirname = pdf_path.split("\\")[-2]
    pdf_name = pdf_path.split("\\")[-1].split(".")[0]
    output_pdf_dirpath = f"{output_folder_path}/{pdf_parent_dirname}/{pdf_name}"

    if os.path.exists(output_pdf_dirpath) and os.path.isdir(output_pdf_dirpath):
        count_files = len(glob.glob(f"{output_pdf_dirpath}/*.{extension}"))
        if count_files > 0 and not overwrite:
            print(f"! Folder {output_pdf_dirpath} already exists with {count_files} files")
            return count_files, output_pdf_dirpath
    else:
        # create folder for the pdf
        os.makedirs(output_pdf_dirpath, exist_ok=True)

    # extract images from pdf
    images = convert_from_path(
        pdf_path, size=resolution, poppler_path=r"poppler-22.04.0/Library/bin"
    )

    # save images use multithreading (map) with tqdm progress bar
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = executor.map(
            lambda i: images[i].save(
                f"{output_pdf_dirpath}/{pdf_name}_{i}.{extension}"
            ) if f"{output_pdf_dirpath}/{pdf_name}_{i}.{extension}" not in glob.glob(f"{output_folder_path}/**/*.{extension}", recursive=True) else None,
            range(len(images))
        )
    return len(images), output_pdf_dirpath


def generate_images(input_folder_path: str, extension: str = "png", resolution: int = None, overwrite : bool = False):
    """Generate images from pdfs in a folder

    Args:
        resolution (int): Resolution of the images
        input_folder_path (str): Path to the folder containing the pdfs. Example: "data/NO_Quad_15"
    """
    print(f"* Input folder: {input_folder_path}\n* Resolution: {resolution}\n* Extension: {extension}\n* Overwrite: {overwrite}")
    # get all pdfs
    pdfs = glob.glob(f"{input_folder_path}/**/*.pdf", recursive=True)
    input_dirname = input_folder_path.split("\\")[-1]
    # create output folder with the same name as the input folder + parameters
    output_dirname = (
        f"{input_dirname}_extracted__{resolution or 'original'}_{extension}"
    )
    output_folder_path = input_folder_path.replace(input_dirname, output_dirname)
    print(f"* Output folder: {output_folder_path}")
    os.makedirs(output_folder_path, exist_ok=True)

    count_images = sum(
        generate_images_from_pdf(
            pdf_path,
            output_folder_path,
            extension=extension,
            resolution=resolution,
            overwrite=overwrite,
        )[0]
        for pdf_path in tqdm(pdfs)
    )
    # total images generated
    print(f"* Total images generated: {count_images}, Total pdfs: {len(pdfs)}")

def main_generate_all_images():
    # generate images from pdfs in a folder
    input_folder_path = "data/NO_Quad_15"
    generate_images(input_folder_path, extension="png", resolution=300, overwrite=False)

if __name__ == "__main__":
    print(get_images_paths())
