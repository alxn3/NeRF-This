from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Dict, Optional, Type

import torch
from nerfstudio.pipelines.dynamic_batch import (DynamicBatchPipeline,
                                                DynamicBatchPipelineConfig)
from tqdm import tqdm
from typing_extensions import Literal

from nerfthis.nerfthis_diffusion import DeNoiseThis


@dataclass
class NerfThisPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for pipeline instantiation"""
    _target: Type = field(default_factory=lambda: NerfThisPipeline)
    max_walk_iterations: int = 3
    edit_rate: int = 100
    edit_count: int = 1
    dnt_device: Optional[str] = None
    use_full_precision: bool = True


class NerfThisPipeline(DynamicBatchPipeline):
    """NerfThis pipeline"""

    config: NerfThisPipelineConfig

    def __init__(
        self,
        config: NerfThisPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.train_num_added_images = len(self.datamanager.train_dataset) // 2
        self.diffusion_indices_order = cycle(range(self.train_num_added_images))
        self.dnt_device = (
            torch.device(device)
            if self.config.dnt_device is None
            else torch.device(self.config.dnt_device)
        )
        self.dnt = DeNoiseThis(self.dnt_device, use_full_precision=self.config.use_full_precision)
        self.diffused_all = False

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        super().load_pipeline(loaded_state, step)
                
        # copy all the original input camera and images to a new list
        half = len(self.datamanager.train_dataset) // 2

        torch.cuda.empty_cache()
        if self.training:
            print("Finding new camera positions...")
            for i in tqdm(range(half)):
                for _ in range(self.config.max_walk_iterations):

                    current_ray_bundle = self.datamanager.train_dataset.cameras.to(self.device).generate_rays(i, keep_shape=True)
                    # reduce ray bundle size by a factor of 2 
                    current_ray_bundle = current_ray_bundle[::2, ::2]

                    camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)

                    accumulation = camera_outputs["accumulation"].unsqueeze(dim=0)
                    accum_sum = torch.sum(accumulation[accumulation > 0.8])
                    dim = accumulation.size()
                    accum_ratio = accum_sum / torch.prod(torch.tensor(dim))

                    if accum_ratio < 0.8:
                        break

                    depth = camera_outputs["depth"].unsqueeze(dim=0)

                    del accumulation
                    del camera_outputs
                    del current_ray_bundle

                    camera_to_world = self.datamanager.train_dataset.cameras.camera_to_worlds[i].to(self.device)

                    # Calculate normal vector of the center of the density map
                    centerx = depth.shape[1] // 2
                    centery = depth.shape[2] // 2
                    depth_center = depth[:, centerx, centery, 0]
                    dx = depth[:, centerx + 1, centery, 0] - depth[:, centerx - 1, centery, 0]
                    dy = depth[:, centerx, centery + 1, 0] - depth[:, centerx, centery - 1, 0]
                    normal_map_center = -torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), torch.ones_like(dx.unsqueeze(1))], dim=1)
                    normal_map_center = normal_map_center / torch.norm(normal_map_center, dim=1, keepdim=True)

                    # Calculate normal vector of the center of the camera
                    camera_position = -torch.matmul(torch.inverse(camera_to_world[:, :3]), camera_to_world[:, 3:4])
                    normal_camera_center = camera_position - depth_center
                    normal_camera_center = normal_camera_center / torch.norm(normal_camera_center, dim=1, keepdim=True)
                    normal_camera_center = torch.transpose(normal_camera_center, 0, 1)

                    # Calculate the rotation matrix between the two normal vectors
                    rotation_axis = torch.cross(normal_map_center, normal_camera_center, dim=1)
                    rotation_axis = rotation_axis / torch.norm(rotation_axis, dim=1, keepdim=True)
                    angle = torch.acos(torch.clamp(torch.sum(normal_map_center * normal_camera_center, dim=1), -1.0, 1.0))

                    cos_angle = torch.cos(angle).unsqueeze(0)
                    sin_angle = torch.sin(angle).unsqueeze(0)
                    iol = torch.sub(1, cos_angle)

                    rotation_matrix = torch.zeros((3, 3)).to(self.device)
                    rotation_matrix[0, 0] = torch.add(cos_angle, rotation_axis[:, 0]**2 * iol).squeeze()
                    rotation_matrix[0, 1] = (rotation_axis[:, 0] * rotation_axis[:, 1] * iol - rotation_axis[:, 2] * sin_angle).squeeze()
                    rotation_matrix[0, 2] = (rotation_axis[:, 0] * rotation_axis[:, 2] * iol + rotation_axis[:, 1] * sin_angle).squeeze()
                    rotation_matrix[1, 0] = (rotation_axis[:, 1] * rotation_axis[:, 0] * iol + rotation_axis[:, 2] * sin_angle).squeeze()
                    rotation_matrix[1, 1] = torch.add(cos_angle, rotation_axis[:, 1]**2 * iol).squeeze()
                    rotation_matrix[1, 2] = (rotation_axis[:, 1] * rotation_axis[:, 2] * iol - rotation_axis[:, 0] * sin_angle).squeeze()
                    rotation_matrix[2, 0] = (rotation_axis[:, 2] * rotation_axis[:, 0] * iol - rotation_axis[:, 1] * sin_angle).squeeze()
                    rotation_matrix[2, 1] = (rotation_axis[:, 2] * rotation_axis[:, 1] * iol + rotation_axis[:, 0] * sin_angle).squeeze()
                    rotation_matrix[2, 2] = torch.add(cos_angle, rotation_axis[:, 2]**2 * iol).squeeze()

                    # Calculate the new camera position
                    move_distance = 0.01  # Change this value for the desired movement
                    new_camera_position = camera_position + move_distance * torch.transpose(normal_map_center, 0, 1)
                    # Apply the rotation matrix and new camera position to the camera's transformation matrix
                    new_rotation = torch.matmul(camera_to_world[:, :3], rotation_matrix.to(self.device))
                    new_camera_to_world = torch.cat([new_rotation, new_camera_position], dim=1)

                    self.datamanager.train_dataset.cameras.camera_to_worlds[i] = new_camera_to_world
                
    
    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        # diffuse all the images at the start so we aren't training on incorrect images
        if not self.diffused_all:
            self.diffused_all = True
            for i in tqdm(range(self.train_num_added_images)):
                current_ray_bundle = self.datamanager.train_dataset.cameras.to(self.device).generate_rays(i, keep_shape=True)
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)

                rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                accum_image = camera_outputs["accumulation"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                del camera_outputs
                del current_ray_bundle
                torch.cuda.empty_cache()

                edited_image = self.dnt.edit_image("", rendered_image.to(self.dnt_device), accum_image.to(self.dnt_device))

                if (edited_image.size() != rendered_image.size()):
                    edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

                self.datamanager.image_batch["image"][i] = edited_image.squeeze().permute(1,2,0)

        if step % self.config.edit_rate == 0:
            for _ in range(self.config.edit_count): self._diffuse_image(next(self.diffusion_indices_order))

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # update the number of rays for the next step
        if "num_samples_per_batch" not in metrics_dict:
            raise ValueError(
                "'num_samples_per_batch' is not in metrics_dict."
                "Please return 'num_samples_per_batch' in the models get_metrics_dict function to use this method."
            )
        self._update_dynamic_num_rays_per_batch(metrics_dict["num_samples_per_batch"])
        self._update_pixel_samplers()

        # add the number of rays
        assert "num_rays_per_batch" not in metrics_dict
        metrics_dict["num_rays_per_batch"] = torch.tensor(self.datamanager.train_pixel_sampler.num_rays_per_batch)

        return model_outputs, loss_dict, metrics_dict
    def _diffuse_image(self, index: int):
        """
        This method will do the masking and inpainting of the image at the given index

        Args:
            index (int): The index of the image to be diffused
        """
        current_ray_bundle = self.datamanager.train_dataset.cameras.to(self.device).generate_rays(index, keep_shape=True)
        camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)

        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        accum_image = camera_outputs["accumulation"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        del camera_outputs
        del current_ray_bundle
        torch.cuda.empty_cache()

        edited_image = self.dnt.edit_image("", rendered_image.to(self.dnt_device), accum_image.to(self.dnt_device))

        if (edited_image.size() != rendered_image.size()):
            edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

        self.datamanager.image_batch["image"][index] = edited_image.squeeze().permute(1,2,0)