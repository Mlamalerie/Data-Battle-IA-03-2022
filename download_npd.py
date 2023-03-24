# web scraping librairies
from bs4 import BeautifulSoup
import requests
# data manipulation librairies
import pandas as pd
import numpy as np
# image processing librairies
import cv2
import re
import os
import time
from glob import glob

from tqdm import tqdm

ROOT_URL = "https://factpages.npd.no/en/wellbore/PageView/Exploration/All"
#%%
def get_content(url: str) -> str:
    """Get content from url

    Args:
        url (str): url

    Returns:
        str: content
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f'Error while fetching data from {url}')

    return response.content

def get_soup(url: str) -> BeautifulSoup:
    """Get soup from url

    Args:
        url (str): url

    Returns:
        BeautifulSoup: soup
    """
    content = get_content(url)
    return BeautifulSoup(content, 'html.parser')

def get_page_ids_for_wellbores():
    global ROOT_URL
    soup = get_soup(ROOT_URL)
    page_ids = {}
    nav_soup = soup.find('nav', {'id': 'navigationContainer'})
    lis = nav_soup.find('ul', {'class': "uk-nav uk-nav-parent-icon uk-nav-sub"}).find_all('li')

    for li in lis:
        link = li.find('a')
        href = link.get('href')
        text = link.text
        page_ids[text] = href.split('/')[-1]

    # keep only elements who match : "{number}/{number}-{number}*"
    page_ids = {k: v for k, v in page_ids.items() if re.match(r'\d+/\d+-\d+[\sa-zA-Z]{0,2}', k)}

    return page_ids

WELLBORE_NAME_ID_MAP = get_page_ids_for_wellbores()

#%%
def get_wellbore_infos(wellbore_name: str) -> dict:
    global ROOT_URL
    result = {}
    wellbore_name = wellbore_name.replace('_', '/')
    wellbore_id = WELLBORE_NAME_ID_MAP.get(wellbore_name)
    if wellbore_id is None:
        raise ValueError(f'No wellbore found for name {wellbore_name}')
    url = f'{ROOT_URL}/{wellbore_id}'
    soup = get_soup(url)

    # Documents – reported by the production licence (period for duty of secrecy expired)

    documents_soup = soup.find('li', {'id': 'documents-reported-by-the-production-licence-(period-for-duty-of-secrecy-expired)'})
    if documents_soup is None:
        raise ValueError(f'No section documents found for wellbore {wellbore_name}')

    # get all pdf links
    pdf_links = documents_soup.find("table").find_all('a')
    pdf_links = [link.get('href') for link in pdf_links]

    # check which is completion log pdf and which is the completion report pdf

    # match [completion log, COMPLETION LOG, Completion Log, Completion_log, completion_log, ...]
    completion_log_pattern = re.compile(r'completion[\s_-]*log', re.IGNORECASE)
    # match [completion report, COMPLETION REPORT, Completion Report, Completion_report, completion_report, ...]
    completion_report_pattern = re.compile(r'completion[\s_-]*report', re.IGNORECASE)


    for pdf_link in pdf_links:
        if completion_log_pattern.search(pdf_link):
            result['completion_log_pdf'] = pdf_link
        if completion_report_pattern.search(pdf_link):
            result['completion_report_pdf'] = pdf_link

    return result

def download_pdf(url: str, output_dir: str, verbose: bool = True, overwrite: bool = False) -> str:
    """Download pdf from url

    Args:
        url (str): url
        output_dir (str): output directory
    """

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f'Error while fetching data from {url}')

    full_path = os.path.join(output_dir, url.split('/')[-1])
    if os.path.exists(full_path) and not overwrite:
        if verbose:
            print(f'! File {full_path} already exists')
        return full_path

    os.makedirs(output_dir, exist_ok=True)

    with open(full_path, 'wb') as f:
        f.write(response.content)

    if verbose:
        print(f'> File {full_path} downloaded')
    return full_path

def search_completion_pdfs(output_dir: str):

    # match [completion log, COMPLETION LOG, Completion Log, Completion_log, completion_log, ...]
    completion_log_pattern = re.compile(r'completion[\s_-]*log', re.IGNORECASE)
    # match [completion report, COMPLETION REPORT, Completion Report, Completion_report, completion_report, ...]
    completion_report_pattern = re.compile(r'completion[\s_-]*report', re.IGNORECASE)

    # get all pdf files in output_dir with glob
    pdf_files = glob(os.path.join(output_dir, '*.pdf'))

    # check if completion log and completion report already downloaded for wellbore_name
    completion_log_pdf = [f for f in pdf_files if completion_log_pattern.search(f)]
    completion_report_pdf = [f for f in pdf_files if completion_report_pattern.search(f)]

    return completion_log_pdf, completion_report_pdf
def download_completion_pdfs(wellbore_name: str, output_dir: str, verbose: bool = True, overwrite: bool = False) -> dict:
    completions_path = {}
    info = get_wellbore_infos(wellbore_name)
    for pdf_type in ['completion_log_pdf', 'completion_report_pdf']:
        if pdf_type in info:
            try:
                output = download_pdf(info[pdf_type], output_dir, verbose, overwrite)
            except ValueError as e:
                print(f'! {wellbore_name}: {e}')
            else:
                completions_path[pdf_type] = output

        else:
            print(f'! No {pdf_type} found for wellbore {wellbore_name}')

    return completions_path



#%%
# check if completion log and completion report already downloaded for wellbore_name






def main() -> None:
    wellbore_names = os.listdir('data/NO_Quad_15')
    # tqdm is a progress bar, len(wellbore_names) is the total number of iterations
    for wellbore_name in tqdm(wellbore_names, total=len(wellbore_names)):

        new_wellbore_name = wellbore_name.replace('/', '_')
        output_dir = f"data/NO_Quad_15/{new_wellbore_name}"

        # check if completion log and completion report already downloaded for wellbore_name
        completion_log_pdf, completion_report_pdf = search_completion_pdfs(output_dir)
        if len(completion_log_pdf) > 0 and len(completion_report_pdf) > 0:
            print(f'! Completion log and completion report already downloaded for wellbore "{wellbore_name}"')
        else:
            # if not, download them
            download_completion_pdfs(wellbore_name, output_dir, verbose=True, overwrite=False)


if __name__ == '__main__':
    main()