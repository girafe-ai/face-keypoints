import sqlite3
from collections import defaultdict
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


class DatasetAFLW(GenericDataset):
    """`DatasetAFLW <https://ieeexplore.ieee.org/abstract/document/6130513>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``AFLW/flickr`` with ``0`` / ``2`` / ``3`` subdirectories exists.
        split (str): split or splits to be returned. Can be a string or tuple of strings.
            Default: ('train', 'valid', 'test').
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        subset (str, optional): flickr.
        download_url (str, optional): url to download zip file with Dataset.
    """
    base_folder = "AFLW"
    sql_file_name = "aflw.sqlite"
    empty_landmark_value = 0

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subset: Optional[str] = "flickr",
        download_url: Optional[str] = "",
    ) -> None:
        # connect to SQL
        sql_file = Path(root) / self.base_folder / self.sql_file_name
        conn = sqlite3.connect(sql_file)
        conn.row_factory = sqlite3.Row
        self.sql_connection = conn.cursor()

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            subset=subset,
            download_url=download_url
        )

        faces_in_file = defaultdict(list)
        for face in self.data:
            faces_in_file[face["filepath"]].append(face["face_id"])
        self.faces_in_file = faces_in_file

    def _get_data(self) -> List[sqlite3.Row]:
        """
        Get Faces info from database.
        """
        faces_on_image_query = """
        SELECT
            faceimages.filepath,
            faces.face_id,
            facerect.x,
            facerect.y, 
            facerect.w, 
            facerect.h
        FROM
            faces, facerect, faceimages
        WHERE
            faces.file_id = faceimages.file_id and
            faces.face_id = facerect.face_id;
        """
        return self.sql_connection.execute(faces_on_image_query).fetchall()

    def _get_target(self) -> Dict[str, Any]:
        """
        Facial Landmarks coordinates on image. Structured as Dict[`face_id`, List[int]].
        """
        face_details_query = """
        SELECT
            featurecoords.face_id,
            featurecoords.feature_id,
            featurecoords.x,
            featurecoords.y
        FROM
            featurecoords;
        """
        feature_data = self.sql_connection.execute(face_details_query).fetchall()

        # target_data dict of `face_id` with facial landmarks
        target_data = defaultdict(
            lambda: np.array([(self.empty_landmark_value, self.empty_landmark_value)] * 21)
        )
        for row in feature_data:
            # feature_id starts from 1, let's start from 0 instead.
            target_data[row["face_id"]][row["feature_id"] - 1] = (row["x"], row["y"])

        return target_data

    def _replace_not_positive_target(self, target: np.ndarray) -> np.ndarray:
        """
        Replace not positive coordinates with `self.empty_landmark_value`.
        """
        target[target <= 0] = self.empty_landmark_value
        return target

    def _get_target_item(self, item: int) -> np.ndarray:
        data_item = self.data[item]
        data_path = data_item["filepath"]
        target = self.target[data_item["face_id"]]
        if self._is_multiple_faces_on_image(data_path):
            # correct coordinates for cropped image
            face_x = data_item["x"]
            face_y = data_item["y"]
            target -= [face_x, face_y]
        return self._replace_not_positive_target(target)

    def _get_data_item(self, item: int) -> Image.Image:
        data_item = self.data[item]
        data_path = data_item["filepath"]
        image = Image.open(self.files_path / data_path)
        if self._is_multiple_faces_on_image(data_path):
            # should crop an image
            image_h, image_w = image.size

            # Error correction
            face_x = data_item["x"] if data_item["x"] >= 0 else 0
            face_y = data_item["y"] if data_item["y"] >= 0 else 0
            face_w = data_item["w"] if data_item["w"] <= image_w else image_w
            face_h = data_item["h"] if data_item["h"] <= image_h else image_h

            return image.crop((face_x, face_y, face_x + face_w, face_y + face_h))

        return image

    def _is_multiple_faces_on_image(self, data_path: str) -> bool:
        """
        Multiple Faces on one Image.

        If True, image should be cropped and landmarks should be recalculated.
        """
        return len(self.faces_in_file.get(data_path)) > 1
