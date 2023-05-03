import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.decorators import check_main_thread
from rich.console import Console

CONSOLE = Console(width=120)


@dataclass
class NerfThisTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: NerfThisTrainer)
    double: bool = True

class NerfThisTrainer(Trainer):
    def __init__(self, config: NerfThisTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)
    
    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir: Path = self.config.load_dir
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")

            if self.config.double:
                loaded_state["pipeline"]["_model.field.embedding_appearance.embedding.weight"] = torch.cat([loaded_state["pipeline"]["_model.field.embedding_appearance.embedding.weight"]]*2, dim=0)
                loaded_state["optimizers"]["fields"]["state"][0]["exp_avg"] = torch.cat([loaded_state["optimizers"]["fields"]["state"][0]["exp_avg"]]*2, dim=0)
                loaded_state["optimizers"]["fields"]["state"][0]["exp_avg_sq"] = torch.cat([loaded_state["optimizers"]["fields"]["state"][0]["exp_avg_sq"]]*2, dim=0)
            
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()