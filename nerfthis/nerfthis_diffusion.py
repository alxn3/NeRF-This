from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from PIL import Image
from rich.console import Console
from torch import nn
from torchtyping import TensorType
from torchvision import transforms as T

CONSOLE = Console(width=120)

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

INPAINT_SOURCE = "runwayml/stable-diffusion-inpainting"

def dummy(images, **kwargs):
    return images, False
class DeNoiseThis(nn.Module):
    
    def __init__(self,  device: Union[torch.device, str], num_train_timesteps: int = 1000, use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.use_full_precision = use_full_precision
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(INPAINT_SOURCE, 
                                                              solver_p=UniPCMultistepScheduler,
                                                              revision="fp16", 
                                                              torch_dtype=torch.float16)
        assert self.pipe is not None
        self.pipe = self.pipe.to(self.device)

        # improve memory performance
        self.pipe.enable_attention_slicing()
        
        self.pipe.safety_checker = dummy
        self.scheduler = self.pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        self.pipe.unet.eval()
        self.pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.use_full_precision:
            self.pipe.unet.float()
            self.pipe.vae.float()
        else:
            if self.device.index:
                self.pipe.enable_model_cpu_offload(self.device.index)
            else:
                self.pipe.enable_model_cpu_offload(0)

        self.unet = self.pipe.unet
        self.auto_encoder = self.pipe.vae
        
    def edit_image(
        self,
        prompt: str,
        image: TensorType["BS", 3, "H", "W"],
        image_accum: TensorType["BS", 1, "H", "W"]
    ) -> torch.Tensor:
        device = image.device
        _, _, height, width = image.shape
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_accum = image_accum.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_accum = np.float32(image_accum)
        image_accum = cv2.GaussianBlur(image_accum, (5, 5), 0)
        mask = self.mask_this(image_accum, "ACC")
        if np.sum(mask) < 25500 or np.sum(mask) > 254*image_accum.shape[0]*image_accum.shape[1]:
            maskcreated = self.create_mask(image, 10)
            maskcreated = np.float32(maskcreated)
            mask = self.mask_this(maskcreated, "SD")
        
        mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((960, 512))
        image = Image.fromarray((image * 255).astype(np.uint8)).resize((960, 512))
        
        prompt = ""
        images = self.pipe(prompt, image, mask_image=mask, guidance_scale=0, num_inference_steps=10, num_images_per_prompt=2, height=512, width=960).images
        arr=np.zeros((height, width, 3), np.float32)
        N = len(images)
        for im in images:
            imarr = np.array(im.resize((width, height)), dtype=np.float32)
            arr = arr+imarr/N
        arr = np.array(np.round(arr), dtype=np.uint8)

        return T.ToTensor()(arr).to(device).unsqueeze(0)
    def create_mask(self, image, threshold):
        # Grayscale the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        # Threshold the result to create a binary mask
        _, mask = cv2.threshold(np.abs(laplacian), threshold, 255, cv2.THRESH_BINARY)
        # Convert to uint8
        mask = cv2.convertScaleAbs(mask)
        # Display the mask
        return np.array(mask)
    
    def mask_this(self, image_accum, pipe="ACC"):
        if pipe == "ACC":
            threshold_value = 250
            _, invertFoxAccPic = cv2.threshold(image_accum, threshold_value, 255, cv2.THRESH_BINARY_INV)
        else:
            invertFoxAccPic = image_accum
        pixels = invertFoxAccPic.reshape((-1, 1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        K = 2
        attempts = 10
        _, labels, _ = cv2.kmeans(pixels, K, None, criteria, attempts, flags)
        mask = labels.reshape(invertFoxAccPic.shape)
        mask = np.uint8(mask)

        # Create a convex mask that wraps all of the white parts of the image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw all contours
        contoursPic = cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)
        # display the image
        hull = []
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i]))
        mask_convex = np.zeros_like(mask)
        cv2.drawContours(mask_convex, hull, -1, 255, -1)

        # feather the edges of the mask massively
        mask_convex = cv2.blur(mask_convex, (200, 200))

        threshold_value = 10
        _, mask_convex = cv2.threshold(mask_convex, threshold_value, 255, cv2.THRESH_BINARY)
        mask_convex = cv2.blur(mask_convex, (200, 200))

        if pipe == "ACC":
        # feather the edges of the mask massively
            threshold_value = 10
            _, mask_convex = cv2.threshold(mask_convex, threshold_value, 255, cv2.THRESH_BINARY)
            threshold_value = 10
            _, mask_convex = cv2.threshold(mask_convex, threshold_value, 255, cv2.THRESH_BINARY)
            
            threshold_value = 10
            _, mask_convex = cv2.threshold(mask_convex, threshold_value, 255, cv2.THRESH_BINARY_INV)
        return mask_convex
    
    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    
    