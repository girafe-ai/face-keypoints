from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from catalyst.contrib.datasets.cifar import VisionDataset

from .utils import download_and_extract_archive


class GenericDataset(VisionDataset):
    base_folder = ''
    split_seed = 42
    split_proportions = {"train": 0.5, "test": 0.2, "validate": 0.3}

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subset: str = "01_Indoor",
        download_url: str = "",
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.subset = subset
        self.split = split
        self.files_path = Path(self.root) / self.base_folder / self.subset

        if not self._check_integrity() and not self._download(download_url):
            raise Exception(
                "You should provide correct `download_url` or put Dataset files in root folder '%s', "
                "this path is empty '%s'" % (root, self.files_path)
            )

        self.data = self._get_data()
        self.target = self._get_target()
        self.files_length = len(self.data)

        self.split_indexes = self._get_split(
            self.files_length,
            self.split_proportions,
        )[self.split]

    def _get_target(self) -> Union[List[Any], Dict[str, Any]]:
        raise NotImplementedError()

    def _get_data(self) -> List[any]:
        raise NotImplementedError()

    def _get_target_item(self, item):
        raise NotImplementedError()

    def _get_data_item(self, item):
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            Dict: (image, landmarks) where landmarks have all the facial points.
        """
        item = self.split_indexes[index]

        image = self._get_data_item(item)

        target = self._get_target_item(item)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": image, "landmarks": target}

    def _get_split(
        self,
        length: int,
        splits: Dict[str, float],
    ) -> Dict[str, list]:
        """
        Creates named splits of indexes (0 to length) proportional to given fractions
        in splits.values().

        Args:
            length: Length of indices to split
            splits: Dictionary of split names and their ratios.

        Returns:
            Dictionary of named splits.
        """
        fracs = np.array(list(splits.values()))
        fracs = fracs / fracs.sum()
        sections = (fracs.cumsum() * length).astype(int)[:-1]
        np.random.seed(self.split_seed)
        indexes = np.random.permutation(length)
        parts = np.split(indexes, sections)
        return {name: data.tolist() for name, data in zip(splits, parts)}

    def __len__(self) -> int:
        return self.files_length

    def _check_integrity(self) -> bool:
        return self.files_path.exists()

    def _read_pts(self, filename):
        return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

    def extra_repr(self) -> str:
        return f"Split: {self.split}"

    def _download(self, download_url: str) -> bool:
        if not download_url:
            return False

        download_and_extract_archive(
            download_url, self.root
        )
        if not self._check_integrity():
            return False
        return True


class Dataset300W(GenericDataset):
    """`Dataset300W <https://ibug.doc.ic.ac.uk/resources/300-W/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``300W`` exists.
        split (str): split or splits to be returned. Can be a string or tuple of strings.
            Default: ('train', 'valid', 'test').
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        subset (str, optional): 01_Indoor / 02_Outdoor.
        download_url (str, optional): url to download zip file with Dataset.
    """
    base_folder = '300W'

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subset: Optional[str] = "01_Indoor",
        download_url: Optional[str] = "",
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            subset=subset,
            download_url=download_url
        )

    def _get_data(self) -> List[Any]:
        return list(self.files_path.glob("*.png"))

    def _get_target(self) -> List[Any]:
        return list(self.files_path.glob("*.pts"))

    def _get_target_item(self, item: int) -> np.ndarray:
        data_path = self.data[item]
        target_path = data_path.with_suffix(".pts")
        return self._read_pts(target_path)

    def _get_data_item(self, item: int) -> Image.Image:
        data_path = self.data[item]
        return Image.open(data_path)
