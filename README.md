# NeRF This!

## Preresiquites
1. If on WSL, Make sure CUDA Toolkit 11.7.1 is installed.
2. Set up Nerfstudio environment
   ```sh
    conda create --name nerfstudio -y python=3.10
    conda activate nerfstudio
    python -m pip install --upgrade pip
    conda install --channel "nvidia/label/cuda-11.7.1" cuda -y
    pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install git+https://github.com/KAIR-BAIR/nerfacc.git
    git clone https://github.com/nerfstudio-project/nerfstudio.git
    cd nerfstudio
    pip install --upgrade pip setuptools
    pip install -e .
    cd ..
    ```
3. Clone this repo and install
   ```sh
    git clone ...
    cd 
    pip install -e .
   ```

## Running

Please follow the instructions [here](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html) to run the model.
- Using the "Resume from checkpoint section", load a trained NeRF using instruct-ngp into the nerfthis model to train.

Example:
```sh
ns-train nerfthis --data data/fox/ --load-dir outputs/fox/instant-ngp/{...}/nerfstudio_models/
```

## Current Restrictions
- Training from checkpoints do not work currently
  - New camera views are being made from new accumulation.