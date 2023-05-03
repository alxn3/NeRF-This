from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfthis.nerfthis import NerfThisModelConfig
from nerfthis.nerfthis_datamanager import NerfThisDataManagerConfig
from nerfthis.nerfthis_dataparser import NerfThisDataParserConfig
from nerfthis.nerfthis_pipeline import NerfThisPipelineConfig
from nerfthis.nerfthis_trainer import NerfThisTrainerConfig

nerfthis_method = MethodSpecification(
    config=NerfThisTrainerConfig(
        method_name="nerfthis",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=NerfThisPipelineConfig(
            max_num_samples_per_ray=1024,
            model=NerfThisModelConfig(
                eval_num_rays_per_chunk=1 << 13,
            ),
            datamanager=NerfThisDataManagerConfig(
                dataparser=NerfThisDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                double=True,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRF-This!"
)
