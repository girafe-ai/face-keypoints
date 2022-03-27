import os
from typing import Dict

import gdown
import numpy as np
from catalyst.contrib.datasets.misc import (
    _extract_archive,
    _check_integrity,
    _gen_bar_updater
)


def get_split(
    length: int,
    splits: Dict[str, float],
    random: bool = False,
    seed: int = 42,
) -> Dict[str, list]:
    """Creates named splits of indexes (0 to length) proportional to given fractions
    in splits.values().

    Args:
        length: Length of indices to split
        splits: Dictionary of split names and their ratios.
        random: make random unordered splits, otherwise sequential split
        seed: set a random seed

    Returns:
        Dictionary of named splits.
    """
    fracs = np.array(list(splits.values()))
    fracs = fracs / fracs.sum()
    sections = (fracs.cumsum() * length).astype(int)[:-1]
    if random:
        np.random.seed(seed)
        inds = np.random.permutation(length)
    else:
        inds = np.arange(length)
    parts = np.split(inds, sections)
    return {name: data.tolist() for name, data in zip(splits, parts)}


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Copied from `catalyst.contrib.datasets.misc` to support Google Disc urls.

    Args:
        url: URL to download file from
        root: Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
            If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download.
            If None, do not check

    Raises:
        IOError: if failed to download url
        RuntimeError: if file not found or corrupted
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if _check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            _, header = urllib.request.urlretrieve(url, fpath, reporthook=_gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                _, header = urllib.request.urlretrieve(url, fpath, reporthook=_gen_bar_updater())
            else:
                raise e

        if header.get_content_type() == "text/html" and "drive.google.com" in url:
            gdown.download(url, fpath)

        # check integrity of downloaded file
        if not _check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def download_and_extract_archive(
    url, download_root, extract_root=None, filename=None, md5=None, remove_finished=False
):
    """
    Copied from `catalyst.contrib.datasets.misc` to overwrite `download_url`.

    :param url:
    :param download_root:
    :param extract_root:
    :param filename:
    :param md5:
    :param remove_finished:
    :return:
    """
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    _extract_archive(archive, extract_root, remove_finished)
