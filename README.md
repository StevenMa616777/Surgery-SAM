# Surgery—SAM

## Overview
This project implements a deep learning model for medical image segmentation, specifically designed for surgical images. It utilizes the `SurgerySAM` model, a variant of SAM(Segment Anything Model), for segmenting medical images with high precision. The implementation is based on PyTorch and is designed to be run on GPU-enabled machines.

## Installation

### Prerequisites
- Python 3.x
- PyTorch (compatible with CUDA for GPU usage)
- Other dependencies listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/StevenMa616777/Surgery-SAM.git
   cd Surgery-SAM/
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the model, run the training script with the desired parameters:
  ```bash
  python train_model.py --num_epochs 200 --learning_rate 1e-5 --batch_size 8 --warmup_steps 500 --num_classes 7
  ```
### Monitoring Training
Use TensorBoard to monitor the training progress:
  ```bash
  tensorboard --logdir=/tf_logs/runs/
  ```


   
