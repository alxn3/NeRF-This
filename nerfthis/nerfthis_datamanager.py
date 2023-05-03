from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from rich.console import Console

from nerfthis.nerfthis_dataparser import NerfThisDataParserConfig

CONSOLE = Console(width=120)

from typing import Union

import torch
from typing_extensions import Literal


@dataclass
class NerfThisDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the NerfThisDataManager."""

    _target: Type = field(default_factory=lambda: NerfThisDataManager)
    dataparser = NerfThisDataParserConfig()
    """config for if we need to double the input images for training (only False if resuming from checkpoint)"""
    double: bool = True

class NerfThisDataManager(VanillaDataManager):
    """Data manager for NerfThisDataManagerConfig."""

    config: NerfThisDataManagerConfig

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
    def setup_train(self):
        super().setup_train()

        # preload the image_batch
        self.image_batch = next(self.iter_train_image_dataloader)

