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

from factpages_npd import WELL_NAME_PAGE_ID_MAP, get_well_general_infos, get_well_extra_infos, search_completion_pdfs, \
    download_completion_pdfs

from tools import generate_images_from_pdf

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
st.sidebar.markdown("Made by [Mlamali SAID SALIMO](), [Thomas Danguilhen]() and [Lionel OBAME]()")
st.sidebar.caption("Look ... #todo ... [here](https://blog.streamlit.io/create-a-color-palette-from-any-image/).")
st.sidebar.markdown("---")
st.sidebar.success("This is a **beta** version of the app.")


# =======
#   Utils
# =======


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


with gallery_tab:
    options = WELL_NAME_PAGE_ID_MAP.keys()
    with st.form("my_form"):
        well_name = st.selectbox("Wellbore name", options, index=1000)

        submitted = st.form_submit_button("Submit")
        st.session_state.well_name = well_name
        st.session_state.submitted = submitted

        # Process the form
        if submitted:

            # get general infos
            with st.expander("ℹ️ - General Information"):
                general_info = get_well_general_infos(well_name)
                df_general_info = update_general_info(well_name)
                st.table(df_general_info)

            # get completion pdfs
            with st.expander("📄 - Completion logs and reports"):
                # find pdf files : completion logs and reports
                well_dir_output = f"{PDFS_DATABASE_PATH}/{well_name.replace('/', '_')}"

                completion_log_pdf_, completion_report_pdf_ = search_completion_pdfs(well_dir_output)
                completion_log_pdf = completion_log_pdf_[0] if completion_log_pdf_ else None
                completion_report_pdf = completion_report_pdf_[0] if completion_report_pdf_ else None
                if completion_log_pdf and completion_report_pdf:
                    st.success("Completion log and completion report already downloaded")
                    for pdf in [completion_log_pdf, completion_report_pdf]:
                        st.markdown(f" - [{os.path.basename(pdf)}]({pdf})")

                if not completion_log_pdf or not completion_report_pdf:
                    st.warning("No completion log or report found")
                    # download pdf files
                    with st.spinner("Downloading completion log and report from NPD website"):
                        pdfs_dict = download_completion_pdfs(well_name, well_dir_output, verbose=True, overwrite=False)

                    completion_log_pdf = pdfs_dict.get("completion_log_pdf")
                    completion_report_pdf = pdfs_dict.get("completion_report_pdf")

                    if completion_log_pdf and completion_report_pdf:
                        st.success("Completion log and completion report downloaded", icon="✅")
                    else:
                        st.error(
                            "Completion log and completion report not found... Please choise another wellbore name...",
                            icon="❌")

            with st.expander("🖼️ - Images from pdf files"):
                well_images_dir_output = well_dir_output.replace("NO_Quad_15", "NO_Quad_15_extracted__original_png")
                with st.spinner("Generating images from pdf files, please wait..."):
                    # generate images from pdf files
                    st.warning(f"Output folder: '{well_images_dir_output}'")
                    for pdf_path in [completion_log_pdf, completion_report_pdf]:
                        if pdf_path:
                            nb_images_generated = generate_images_from_pdf(pdf_path=pdf_path,
                                                                           output_folder_path=well_images_dir_output,
                                                                           extension="png", overwrite=False)
                            st.success(f"{nb_images_generated} images generated from {os.path.basename(pdf_path)}",
                                       icon="🖼️")

            # process images with pattern matching, get data

            # process images with OCR, get data

            # combine data from pattern matching and OCR -> display pie chart
            st.write("## - Pie chart -")
            # Créer un graphique en camembert avec des valeurs aléatoires
            values = [random.randint(1, 10) for _ in range(4)]
            fig = px.pie(values=values, names=["A", "B", "C", "D"])
            st.plotly_chart(fig)

# show the image
# with st.expander("🖼  Artwork", expanded=True):
#    st.image(img, use_column_width=True)
