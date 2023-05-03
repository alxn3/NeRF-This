# NeRF This!

## Preresiquites
1. If on WSL, Make sure CUDA Toolkit 11.7.1 is installed.
   - Make sure that you do `sudo ln -s /usr/lib/wsl/lib/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so` and paths
      ```sh
      export PATH="/usr/local/cuda/bin:$PATH"
      export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
      ```
      have been set so that compilers can find `-lcuda` during setup.
2. Set up Nerfstudio environment
   ```sh
   conda create --name nerfstudio -y python=3.10
   conda activate nerfstudio
   python -m pip install --upgrade pip
   conda install --channel "nvidia/label/cuda-11.7.1" cuda -y
   pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
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
   git clone https://github.com/alxn3/NeRF-This.git
   cd NeRF-This
   pip install -e .
   ```

## Running

Please follow the instructions [here](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html) to run the model.
- Using the "Resume from checkpoint section", load a trained NeRF using instruct-ngp into the nerfthis model to train.

Example:
```sh
# Train a NeRF model first
ns-train instant-ngp --data data/fox/

# Run our method on the trained model
ns-train nerfthis --data data/fox/ --load-dir outputs/fox/instant-ngp/{...}/nerfstudio_models/
```

Please note that the step count starts at what the trained model ended at. For example, if the trained model stopped at 30k steps, our model will start from that step as well.

## Current Restrictions
- Training from checkpoints do not work currently
  - New camera views are being made from new accumulation.


## Compute Requirements
- 16 GB VRAM
  - Note: will occasionally throw `CUDA_ERROR_OUT_OF_MEMORY` errors during the setup phase.

Training 200k steps took ~8 hours using:
- RTX Quadro 5000 16 GB
- Intel Xeon W-10885M
- 32 GB RAM