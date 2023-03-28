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


# %%
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
    """Get page ids for wellbores. Example: 15
    
    Returns:
        dict: {well_name: page_id}

    """
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


WELL_NAME_PAGE_ID_MAP = get_page_ids_for_wellbores()  # {well_name: page_id}


# %%
def get_well_general_infos(well_name: str) -> dict:
    """Get well general infos.
    
    Args:
        well_name (str): well name. Example: "15/2-1"
            
    Returns:
        pd.DataFrame: well general infos
    """
    global ROOT_URL
    well_name = well_name.replace('_', '/')
    well_page_id = WELL_NAME_PAGE_ID_MAP.get(well_name)
    if well_page_id is None:
        raise ValueError(f'No wellbore found for name {well_name}')
    url = f'{ROOT_URL}/{well_page_id}'

    soup = get_soup(url)

    # - General information
    general_info_soup = soup.find('li', {'id': 'general-information'})
    if general_info_soup is None:
        raise ValueError(f'No section general information found for wellbore {well_name}')

    # get all rows
    rows = general_info_soup.find("table").find("tbody").find_all('tr')
    # get all keys, values  : key = first element of row, value = second element of row
    keys = []
    values = []
    for row in rows:
        contents = [content for content in row.contents if content != '\n']
        # todo : check if key 'Reclassified to wellbore' is present
        key = contents[0].contents[0] if len(contents[0].contents) > 0 else contents[0].text
        value = contents[-1].text
        # remove all \n and extra spaces (replace by one space)
        key = re.sub(r'\s+', ' ', key).strip()
        value = re.sub(r'\s+', ' ', value).strip()
        keys.append(key)
        values.append(value)

    return dict(zip(keys, values))


tmp = get_well_general_infos('9/2-7 S')


# %%


def get_well_extra_infos(well_name: str) -> dict:
    global ROOT_URL
    result = {}
    well_name = well_name.replace('_', '/')
    well_page_id = WELL_NAME_PAGE_ID_MAP.get(well_name)
    if well_page_id is None:
        raise ValueError(f'No wellbore found for name {well_name}')
    url = f'{ROOT_URL}/{well_page_id}'
    soup = get_soup(url)

    # Documents – reported by the production licence (period for duty of secrecy expired)

    documents_soup = soup.find('li', {
        'id': 'documents-reported-by-the-production-licence-(period-for-duty-of-secrecy-expired)'})
    if documents_soup is None:
        raise ValueError(f'No section documents found for wellbore {well_name}')

    # get all pdf links
    pdf_links = documents_soup.find("table").find_all('a')
    pdf_links = [link.get('href') for link in pdf_links]

    # check which is completion log pdf and which is the completion report pdf

    # match [completion log, COMPLETION LOG, Completion Log, Completion_log, completion_log, ...]
    completion_log_pattern = re.compile(r'(completion[\s_-]*log)|(composite[\s_-]*well[\s_-]*log)', re.IGNORECASE)
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


def search_completion_pdfs(where_dir: str):
    """Search for completion log and completion report pdfs in where_dir

    Args:
        output_dir (str): output directory

    Returns:
        tuple: paths to completion log and completion report pdfs
    """

    # match [completion log, COMPLETION LOG, Completion Log, Completion_log, completion_log, ..., Composite_well_log, composite_well_log, ...]
    completion_log_pattern = re.compile(r'(completion[\s_-]*log)|(composite[\s_-]*well[\s_-]*log)', re.IGNORECASE)
    # match [completion report, COMPLETION REPORT, Completion Report, Completion_report, completion_report, ...]
    completion_report_pattern = re.compile(r'completion[\s_-]*report', re.IGNORECASE)

    # get all pdf files in output_dir with glob
    pdf_files = glob(os.path.join(where_dir, '*.pdf'))

    # check if completion log and completion report already downloaded for well_name
    completion_log_pdf = [f for f in pdf_files if completion_log_pattern.search(f)]
    completion_report_pdf = [f for f in pdf_files if completion_report_pattern.search(f)]

    return completion_log_pdf, completion_report_pdf


def download_completion_pdfs(well_name: str, output_dir: str, verbose: bool = True,
                             overwrite: bool = False) -> dict:
    """Download completion log and completion report pdfs for well_name

    Args:
        well_name (str): well name
        output_dir (str): output directory
        verbose (bool, optional): verbose. Defaults to True.
        overwrite (bool, optional): overwrite. Defaults to False.

    Returns:
        dict: dict with keys 'completion_log_pdf' and 'completion_report_pdf' and values the paths to the pdfs
    """
    completions_path = {}
    info = get_well_extra_infos(well_name)
    for pdf_type in ['completion_log_pdf', 'completion_report_pdf']:
        if pdf_type in info:
            try:
                output = download_pdf(info[pdf_type], output_dir, verbose, overwrite)
            except ValueError as e:
                print(f'! {well_name}: {e}')
            else:
                completions_path[pdf_type] = output

        else:
            print(f'! No {pdf_type} found for wellbore {well_name}')

    return completions_path


# %%
# check if completion log and completion report already downloaded for well_name


def main_download() -> None:
    wellbore_names = os.listdir('data/NO_Quad_15')
    # tqdm is a progress bar, len(wellbore_names) is the total number of iterations
    for well_name in tqdm(wellbore_names, total=len(wellbore_names)):

        new_wellbore_name = well_name.replace('/', '_')
        output_dir = f"data/NO_Quad_15/{new_wellbore_name}"

        # check if completion log and completion report already downloaded for well_name
        completion_log_pdf, completion_report_pdf = search_completion_pdfs(output_dir)
        if len(completion_log_pdf) > 0 and len(completion_report_pdf) > 0:
            print(f'! Completion log and completion report already downloaded for wellbore "{well_name}"')
        else:
            # if not, download them
            try:
                download_completion_pdfs(well_name, output_dir, verbose=True, overwrite=False)
            except ValueError as e:
                print(f'! {well_name}: {e}')


if __name__ == '__main__':
    main_download()
