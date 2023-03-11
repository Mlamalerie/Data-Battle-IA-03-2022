from pdf2image import convert_from_path
import glob


def generate_images(resolution: int, output_folder) -> None:
    pdfs = glob.glob("*/*/*/*.pdf")
    i = 0
    for pdf in pdfs:
        images = convert_from_path(
            pdf, size=resolution, poppler_path=r"poppler-22.04.0/Library/bin"
        )
        for image in images:
            i += 1
            image.save(
                output_folder + "/" + pdf.split("\\")[-2] + "/" + str(i) + ".png"
            )
