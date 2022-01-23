import operator
import os
import shutil
import random
from functools import reduce
from pathlib import Path

import gdown
import pickle
import torch
from scipy.io import loadmat

from typing import Optional, Callable, Any, Tuple, List

from PIL import Image


import numpy as np
import torchvision.datasets
from catalyst.contrib.datasets.cifar import VisionDataset
from catalyst.contrib.datasets.misc import _extract_archive, _check_integrity, _gen_bar_updater
# download_and_extract_archive
from torchtext.datasets import Multi30k
from torchvision.datasets.utils import check_integrity

from torch.utils.data import random_split


import numpy as np
from evopy.types import Dict, Number


RANDOM_SEED = 42


def get_split(
    length: int,
    splits: Dict[str, Number],
    random: bool = False,
) -> Dict[str, list]:
    """Creates named splits of indexes (0 to length) proportional to given fractions
    in splits.values().

    Args:
        length: Length of indices to split
        splits: Dictionary of split names and their ratios.
        random: make random unordered splits, otherwise sequential split

    Returns:
        Dictionary of named splits.
    """
    fracs = np.array(list(splits.values()))
    fracs = fracs / fracs.sum()
    sections = (fracs.cumsum() * length).astype(int)[:-1]
    if random:
        np.random.seed(RANDOM_SEED)
        inds = np.random.permutation(length)
    else:
        inds = np.arange(length)
    parts = np.split(inds, sections)
    return {name: data.tolist() for name, data in zip(splits, parts)}

class GoogleDriveDownloadException(Exception):
    pass


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


# def split_dataset(path_to_dataset, train_ratio, valid_ratio):
#     _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
#     sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)
#
#     # directories where the splitted dataset will lie
#     dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
#     dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
#     dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')
#
#     for i, sub_dir in enumerate(sub_dirs):
#
#         dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
#         dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
#         dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset
#
#         # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
#         class_name = sub_dir
#         sub_dir = os.path.join(path_to_dataset, sub_dir)
#         sub_dir_item_cnt[i] = len(os.listdir(sub_dir))
#
#         items = os.listdir(sub_dir)
#
#         # transfer data to trainset
#         for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
#             if not os.path.exists(dir_train_dst):
#                 os.makedirs(dir_train_dst)
#
#             source_file = os.path.join(sub_dir, items[item_idx])
#             dst_file = os.path.join(dir_train_dst, items[item_idx])
#             shutil.copyfile(source_file, dst_file)
#
#         # transfer data to validation
#         for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
#                               round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
#             if not os.path.exists(dir_valid_dst):
#                 os.makedirs(dir_valid_dst)
#
#             source_file = os.path.join(sub_dir, items[item_idx])
#             dst_file = os.path.join(dir_valid_dst, items[item_idx])
#             shutil.copyfile(source_file, dst_file)
#
#         # transfer data to testset
#         for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
#             if not os.path.exists(dir_test_dst):
#                 os.makedirs(dir_test_dst)
#
#             source_file = os.path.join(sub_dir, items[item_idx])
#             dst_file = os.path.join(dir_test_dst, items[item_idx])
#             shutil.copyfile(source_file, dst_file)
#
#     return


class Dataset300W(VisionDataset):
    """`Dataset300W <https://ibug.doc.ic.ac.uk/resources/300-W/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        split (str, optional): split or splits to be returned. Can be a string or tuple of strings.
            Default: ('train', 'valid', 'test').
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = '300W_LP'
    split_seed = 42
    split_train_val_test_proportions = [0.5, 0.2, 0.3]

    # meta = {
    #     'filename': 'batches.meta',
    #     'key': 'label_names',
    #     'md5': '5ff9c542aee3614f3951f8cda6e48888',
    # }

    def __init__(
            self,
            root: str,
            split: Optional[str] = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            url: str = "",
            subset: str = "AFW",
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.url = url

        # torch.utils.data.random_split()
        self.subset = subset

        self.data_type = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        downloaded_list = self.get_dataset_splitted_part()

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = Path(self.root) / self.base_folder / file_name
            with open(file_path, 'rb') as f:
                pass
                # entry = pickle.load(f, encoding='latin1')
                # self.data.append(entry['data'])
                # if 'labels' in entry:
                #     self.targets.extend(entry['labels'])
                # else:
                #     self.targets.extend(entry['fine_labels'])

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        #
        # self._load_meta()

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
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        path = Path(self.root) / self.base_folder
        if path.exists():
            return True
        return False

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            self.url, self.root, filename="300W-LP.zip"
        )
        # Split and put it in different dirs?
        # self.split_dataset()

    def get_dataset_splitted_part(self) -> List[str]:
        """
        Get requested dataset split data.
        :return: List[str]
        """
        files_path = Path(self.root) / self.base_folder / self.subset
        all_files = list(files_path.glob("*.jpg"))
        files_length = len(all_files)
        random.Random(self.split_seed).shuffle(all_files)

        split = list(
            map(lambda x: round(x * files_length), self.split_train_val_test_proportions)
        )

        result = dict()
        result["train"], result["validate"], result["test"] = np.split(
            np.array(all_files), [split[0], split[0] + split[1]]
        )

        return result[self.data_type]

    def extra_repr(self) -> str:
        return f"Split: {self.data_type}"


# dataset = Dataset300W("./datasets", download=True, url="https://drive.google.com/uc?export=download&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k")
