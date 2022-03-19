import os
from pathlib import Path
from typing import Optional, Callable, Any, Tuple, Dict

import gdown
import numpy as np
from PIL import Image
from catalyst.contrib.datasets.cifar import VisionDataset
from catalyst.contrib.datasets.misc import _extract_archive, _check_integrity, _gen_bar_updater
from scipy.io import loadmat


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
    """@TODO: Docs. Contribution is welcome."""
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    _extract_archive(archive, extract_root, remove_finished)


class Dataset300W(VisionDataset):
    """`Dataset300W <https://ibug.doc.ic.ac.uk/resources/300-W/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        split (str): split or splits to be returned. Can be a string or tuple of strings.
            Default: ('train', 'valid', 'test').
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        url (str, optional):
        subset (str, optional):

    """
    base_folder = '300W_LP'
    split_seed = 42
    split_train_val_test_proportions = {"train": 0.5, "test": 0.2, "validate": 0.3}
    url = "https://drive.google.com/uc?export=download&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k"

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        url: str = "",
        subset: str = "AFW",
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        if url:
            self.url = url

        self.subset = subset

        self.data_type = split

        self.download()

        files_path = Path(self.root) / self.base_folder / self.subset
        self.data = list(files_path.glob("*.jpg"))
        self.target = list(files_path.glob("*.mat"))
        self.files_length = len(self.data)

        self.split_indexes = get_split(
            self.files_length,
            self.split_train_val_test_proportions,
            True,
            self.split_seed
        )[self.data_type]

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     if not check_integrity(path, self.meta['md5']):
    #         raise RuntimeError('Dataset metadata file not found or corrupted.' +
    #                            ' You can use download=True to download it')
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        item = self.split_indexes[index]
        data_path = self.data[item]

        img = Image.open(data_path)

        target_path = data_path.with_suffix(".mat")

        target = loadmat(target_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.files_length

    def _check_integrity(self) -> bool:
        path = Path(self.root) / self.base_folder
        return path.exists()

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            self.url, self.root, filename="300W-LP.zip"
        )

    def extra_repr(self) -> str:
        return f"Split: {self.data_type}"
