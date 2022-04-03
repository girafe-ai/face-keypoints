import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, Any, Tuple

from PIL import Image
from catalyst.contrib.datasets.cifar import VisionDataset
from scipy.io import loadmat

from .utils import get_split, download_and_extract_archive


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


class DatasetAFLW(VisionDataset):
    """`DatasetAFLW <https://ieeexplore.ieee.org/abstract/document/6130513>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``AFLW/flickr`` with ``0`` / ``2`` / ``3`` subdirectories exists.
            You should also put ``aflw.sqlite`` file in root / ``AFLW`` directory.
        split (str): split or splits to be returned. Can be a string or tuple of strings.
            Default: ('train', 'valid', 'test').
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = "AFLW"
    split_seed = 42
    split_train_val_test_proportions = {"train": 0.5, "test": 0.2, "validate": 0.3}
    path = "flickr"

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data_type = split

        files_path = Path(self.root) / self.base_folder / self.path

        if not self._check_integrity(path=files_path):
            raise Exception(
                "You should put Dataset files in root folder '%s', this path is empty '%s'" % (root, files_path)
            )

        self.data = list(files_path.glob("**/*.*"))
        self.files_length = len(self.data)

        self.split_indexes = get_split(
            self.files_length,
            self.split_train_val_test_proportions,
            True,
            self.split_seed
        )[self.data_type]

        # get info from SQL
        sql_file = Path(self.root) / self.base_folder / "aflw.sqlite"
        conn = sqlite3.connect(sql_file)
        sql_connection = conn.cursor()

        face_details_query = """
            SELECT 
                faceimages.filepath, 
                featurecoords.x,
                featurecoords.y,
                featurecoordtypes.code
            FROM 
                faceimages, faces, featurecoords, featurecoordtypes
            WHERE 
                faces.file_id = faceimages.file_id and 
                featurecoords.face_id = faces.face_id and 
                featurecoords.feature_id = featurecoordtypes.feature_id;
        """

        data = sql_connection.execute(face_details_query).fetchall()

        target_data = defaultdict(list)
        for row in data:
            target_data[row[0]].append(row[1:])
        self.target_data = target_data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target has all the facial points.
        """
        item = self.split_indexes[index]
        data_path = self.data[item]

        img = Image.open(data_path)

        file_name = str(Path(data_path.parents[0].name) / data_path.name)

        target = self.target_data[file_name]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.files_length

    def _check_integrity(self, path) -> bool:
        return path.exists()

    def extra_repr(self) -> str:
        return f"Split: {self.data_type}"
