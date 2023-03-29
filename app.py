import streamlit as st
import plotly.express as px
import random
import requests
import os
import pandas as pd

from io import BytesIO
from base64 import b64encode
from glob import glob
from PIL import Image, ImageEnhance

from tools import get_images_paths
from detection import detect, SHAPE_LITHO_DICT, proportion_litho

from factpages_npd import WELL_NAME_PAGE_ID_MAP, get_well_general_infos, get_well_extra_infos, search_completion_pdfs, \
    download_completion_pdfs

from tools import generate_images_from_pdf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PDFS_DATABASE_PATH = "data/NO_Quad_15"

st.set_page_config(
    page_title="NPD Wellbore name",
    page_icon="🔎",
    # layout="wide",
    initial_sidebar_state="expanded",
)

# st.image("logo.jpg")
st.sidebar.title("NPD Wellbore App 📊")
st.sidebar.caption("Tell your data story with style.")
st.sidebar.markdown("Made by [Mlamali SAID SALIMO](), and [Lionel OBAME]()")
# st.sidebar.caption("Look ... #todo ... [here](https://blog.streamlit.io/create-a-color-palette-from-any-image/).")
st.sidebar.markdown("---")
st.sidebar.success("This is a **beta** version of the app.")

# =======
#   Session state
# =======
# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "ok_display_charts" in st.session_state:
    st.session_state.ok_display_charts = False

if not "lithology_data" in st.session_state:
    st.session_state.lithology_data = None


# =======
#   Utils
# =======

@st.cache_data
def get_wellbore_name_options():
    options = WELL_NAME_PAGE_ID_MAP.keys()
    options = sorted(options, key=lambda x: int(x.split("/")[0]))
    return options


def get_pdf_content(filepath):
    with open(filepath, 'rb') as f:
        pdf_bytes = f.read()
    return pdf_bytes


# Fonction pour afficher le fichier PDF sur Streamlit
def embed_pdf_viewer(pdf_bytes):
    pdf_url = f"data:application/pdf;base64,{b64encode(pdf_bytes).decode()}"
    st.markdown(f'<iframe src="{pdf_url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)


# =======
#   App
# =======


with st.expander("Informations"):
    st.info(
        '**Wellbore name** are the official name of wellbore based on NPD guidelines for designation of wells and wellbores.',
        icon="ℹ️")

# provide options to either select an well from the npd database or upload a pdf file
gallery_tab, upload_tab = st.tabs(["NPD Database", "Upload"])


@st.cache_resource(show_spinner=True)
def update_general_info(well_name):
    return pd.DataFrame.from_dict(get_well_general_infos(well_name), orient='index', columns=['Value'])


@st.cache_resource(show_spinner=True)
def update_log_lithology_data(page_path, crop_height, detect_sand, detect_limestone, detect_clay, detect_marl):
    result_log, img_log = detect(one_page_image_completion_log_path, detect_vagues=detect_marl,
                                 detect_circles=detect_sand,
                                 detect_rectangles=detect_limestone, detect_ligns=detect_clay, crop_height=CROP_HEIGHT,
                                 display_image=False)
    lithology_data_ = proportion_litho(result_log)
    return lithology_data_, img_log


with gallery_tab:
    options = get_wellbore_name_options()
    with st.form("my_form"):

        well_name = st.selectbox("Wellbore name", options, index=options.index("15/5-2"))
        # todo :  4 checkboxes on the same line, for choose which infos to display in pie chart
        col1, col2, col3, col4 = st.columns(4)

        submitted = st.form_submit_button("Submit")
        st.session_state.well_name = well_name
        st.session_state.submitted = submitted

        # Process the form
        if submitted:

            # 1. get general infos
            with st.expander("ℹ️ - General Information"):
                general_info = get_well_general_infos(well_name)
                df_general_info = update_general_info(well_name)
                st.table(df_general_info)

            # 2. get completion pdfs
            with st.expander("📄 - Completion logs and reports"):
                # create output folder where to download pdf files
                well_dir_output = f"{PDFS_DATABASE_PATH}/{well_name.replace('/', '_')}"
                # find completion logs and reports
                completion_log_pdf_, completion_report_pdf_ = search_completion_pdfs(well_dir_output)
                completion_log_pdf = completion_log_pdf_[0] if completion_log_pdf_ else None
                completion_report_pdf = completion_report_pdf_[0] if completion_report_pdf_ else None
                if completion_log_pdf and completion_report_pdf:
                    st.success("Completion log and completion report already downloaded")
                    for pdf in [completion_log_pdf, completion_report_pdf]:
                        st.markdown(f" - [{os.path.basename(pdf)}]({pdf})")

                if not completion_log_pdf or not completion_report_pdf:
                    if not completion_log_pdf:
                        st.warning(f"No completion log already download for {well_name}")
                    if not completion_report_pdf:
                        st.warning(f"No completion report already download for {well_name}")

                    # download pdf files
                    with st.spinner("Downloading completion log and report from NPD website"):
                        pdfs_dict = download_completion_pdfs(well_name, well_dir_output, verbose=True, overwrite=False)

                    # get completion log and completion report paths
                    completion_log_pdf = pdfs_dict.get("completion_log_pdf")
                    completion_report_pdf = pdfs_dict.get("completion_report_pdf")

                    if completion_log_pdf:
                        st.info(f"Completion log for {well_name} downloaded", icon="ℹ️")

                    if completion_report_pdf:
                        st.info("Completion report downloaded", icon="ℹ️")

                    # todo: stop if no completion log found
                    if not completion_log_pdf and not completion_report_pdf:
                        st.error(
                            "Completion log and completion report not found on NPD website, please upload them manually",
                            icon="❌")

            # 3. generate images from pdf files
            with st.expander("🖼️ - Images from pdf files"):
                RESOLUTION = 4000
                EXTENSION = "png"
                well_images_dir_output = well_dir_output.replace("NO_Quad_15",
                                                                 f"NO_Quad_15_extracted__{RESOLUTION or 'original'}_{EXTENSION}")
                # full parent dir path
                well_images_parent_dir_output = "/".join(well_images_dir_output.split("/")[:-1])
                with st.spinner("Generating images from pdf files, please wait..."):
                    # generate images from pdf files
                    st.warning(f"Output folder: '{well_images_dir_output}'", icon="📁")
                    if completion_log_pdf:
                        nb_images_generated, output_completion_log_pdf_dirpath = generate_images_from_pdf(
                            pdf_path=completion_log_pdf,
                            output_folder_path=well_images_parent_dir_output,
                            extension="png", overwrite=False)
                        st.info(f"{nb_images_generated} images generated from {os.path.basename(completion_log_pdf)}",
                                icon="🖼️")
                        if nb_images_generated:
                            st.session_state.ok_display_images = True
                    if completion_report_pdf:
                        nb_images_generated, output_completion_report_pdf_dirpath = generate_images_from_pdf(
                            pdf_path=completion_report_pdf,
                            output_folder_path=well_images_parent_dir_output,
                            extension="png", overwrite=False)
                        st.info(
                            f"{nb_images_generated} images generated from {os.path.basename(completion_report_pdf)}",
                            icon="🖼️")

            # 4. Process images : get lithology, get wellbore profile
            # 4.1. Get lithology data from completion log
            images_completion_log = get_images_paths(output_completion_log_pdf_dirpath, extension="png")
            st.write(images_completion_log)
            one_page_image_completion_log_path = images_completion_log[0]
            # CROP_HEIGHT = (10000, 18000)
            CROP_HEIGHT = (10000, 16000)
            lithology_data_,img_log = update_log_lithology_data(page_path=one_page_image_completion_log_path,
                                                        crop_height=CROP_HEIGHT, detect_sand=True,
                                                        detect_limestone=True,detect_marl=True, detect_clay=True)

            st.image(Image.fromarray(img_log), use_column_width=True)

            # 4.2. Get wellbore profile from completion report

            st.session_state.lithology_data = lithology_data_.copy()

            # combine data from pattern matching and OCR -> display pie chart
            if st.session_state.submitted and st.session_state.ok_display_images:
                st.markdown("---")
            st.write("## 📊 Pie chart")
            # Créer un graphique en camembert avec lithology_data
            fig = px.pie(values=list(st.session_state.lithology_data.values()),
                         names=list(st.session_state.lithology_data.keys()))
            st.plotly_chart(fig)

            # display session states vars
            with st.expander("Session state"):
                st.write(st.session_state)

                # show the image
                # with st.expander("🖼  Artwork", expanded=True):
                #    st.image(img, use_column_width=True)
