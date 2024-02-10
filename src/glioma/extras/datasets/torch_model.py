from kedro.io import AbstractDataSet
from os.path import isfile
from typing import Any, Dict
import torch

class TorchLocalModel(AbstractDataSet):
    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        self._filepath = filepath
        default_save_args = {}
        default_load_args = {}

        self._load_args = {**default_load_args, **load_args} if load_args is not None else default_load_args
        self._save_args = {**default_save_args, **save_args} if save_args is not None else default_save_args

    def _load(self):
        checkpoint = torch.load(self._filepath)
        # Assuming you want to return the entire checkpoint
        return checkpoint

    def _save(self, checkpoint: Dict[str, Any]) -> None:
        torch.save(checkpoint, self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return isfile(self._filepath)
