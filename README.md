# Surgery—SAM

## Overview
This project implements a deep learning model for medical image segmentation, specifically designed for surgical images. It utilizes the `SurgerySAM` model, a variant of SAM(Segment Anything Model), for segmenting medical images with high precision. The implementation is based on PyTorch and is designed to be run on GPU-enabled machines.

## Installation

### Prerequisites
- Linux (We tested our codes on Ubuntu 18.04)
- Anaconda
- Python 3.10.11
- Pytorch 2.0.0 **(Pytorch 2+ is necessary)**
- Other dependencies listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/StevenMa616777/Surgery-SAM.git
   cd Surgery-SAM/
   ```
2. Install the required Python packages:
   ```bash
   conda create Surgery_SAM python==3.10.11
   conda activate Surgery_SAM
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


   
