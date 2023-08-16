import os
import json
import numpy as np
from pathlib import Path
from typing import Union


class OneDimArray:
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path).expanduser().resolve()
        self.metadata_path = self._get_metadata_path()
        self.metadata = None

    def _get_metadata_path(self) -> Path:
        metadata_filename = "_".join(self.path.name.split(".")) + "_meta.json"
        return self.path.parent / metadata_filename

    @staticmethod
    def _construct_metadata(array: np.ndarray) -> dict:
        return {"dtype": array.dtype.name, "size": array.size}

    def _update_metadata(self) -> None:
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def _dump_metadata(self) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def dump(self, array: np.ndarray, *, overwrite: bool = False) -> None:
        if not overwrite and os.path.exists(self.path):
            raise FileExistsError(f"File already exists at '{self.path}'. Set `overwrite=True` to force writing.")
        if array.ndim != 1:
            raise ValueError("`array` must be 1D")
        array.tofile(self.path)
        self.metadata = self._construct_metadata(array)
        self._dump_metadata()

    def load(self, offset=0, count=-1) -> np.ndarray:
        self._update_metadata()
        return np.fromfile(self.path, dtype=self.metadata["dtype"], count=count, offset=offset)

    def append(self, array: np.ndarray):
        if not os.path.exists(self.path):
            self.dump(array)
            return
        if array.ndim != 1:
            raise ValueError("`array` must be 1D")
        self._update_metadata()
        dtype = self.metadata["dtype"]
        dtype_new = array.dtype.name
        if dtype != dtype_new:
            raise ValueError(f"attempted to append a {dtype_new} array to a {dtype} array")
        self.metadata["size"] += array.size
        self._dump_metadata()

        with open(self.path, "ab") as f:
            f.write(array.tobytes())

    def delete_end(self, del_count: int):
        if del_count <= 0:
            raise ValueError("`del_count` must be positive")
        self._update_metadata()
        new_size = max(self.metadata["size"] - del_count, 0)
        num_bytes = np.dtype(self.metadata["dtype"]).itemsize * new_size
        with open(self.path, "ab") as f:
            f.truncate(num_bytes)
        self.metadata["size"] = new_size
        self._dump_metadata()


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    array_path = OUTPUT_DIR / "array1"

    arr = OneDimArray(array_path)
    arr.dump(np.arange(10), overwrite=True)
    print(arr.metadata)
    print(arr.load())
    print()

    arr.append(np.arange(5))
    print(arr.metadata)
    print(arr.load())
    print()

    arr.delete_end(5)
    print(arr.metadata)
    print(arr.load())
    print()

    arr.delete_end(12)
    print(arr.metadata)
    print(arr.load())
    print()
