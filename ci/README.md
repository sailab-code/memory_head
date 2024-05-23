
<div align="center">
  
  <div>
  <h1>Memory Head for Pre-Trained Backbones in Continual Learning</h1>
  </div>

  <div>
      Matteo Tiezzi &emsp; Federico Becattini &emsp; Simone Marullo &emsp; Stefano Melacci
  </div>
  <br/>

</div>


This repo contains the PyTorch code for CoLLAs 2024 paper "[Memory Head for Pre-Trained Backbones in Continual Learning](tba)".


CODE REPOSITORY CONTENTS
------------------------
This repository is based upon the [SLCA repository](https://github.com/GengDavid/SLCA).
Additionally, this repository contains:

    mh :                 folder containing the source code of our Memory Heads
    main_wb.py :         experiments runner

## Requirements
We tested the code on PyTorch 1.10 and TorchVision 0.11.1

    timm
    tqdm  
    numpy  
    scipy  
    quadprog  
    POT  

## Pre-trained Models
Please download pre-trained ViT-Base models from [MoCo v3](https://drive.google.com/file/d/1bshDu4jEKztZZvwpTVXSAuCsDoXwCkfy/view?usp=share_link) and [ImaegNet-21K](https://drive.google.com/file/d/1PcAOf0tJYs1FVDpj-7lrkSuwXTJXVmuk/view?usp=share_link) and then put or link the pre-trained models to ```pretrained``` folder

## Data 
CIFAR100 will be dowloaded when the script is launched. 
Follow the instructions on the [SLCA repository](https://github.com/GengDavid/SLCA) to download the other datasets.

## Example command lines 
The best selected hyper-parameters can be found in the Appendix of the paper. In the following, some example command lines
to run the experiments:

### Running a Memory Head on the CIFAR100 dataset
    python main_wb.py --bcb_lrscale=0.01 --beta_k=0.001 --ca_epochs=5 --ca_with_logit_norm=0.1 --dataset=cifar100_224 --delta=1 --distance=cosine --epochs=20 --freeze_keys_ca=True --gamma_alpha=1 --increment=10 --init_cls=10 --key_mem_units=10 --layer_norm=False --lr=0.01 --milestones=18 --model_postfix=20e --seed=1996 --tau_alpha=0.95 --tau_eta=15000 --tau_mu=500 --batch_size=8  --virtual_batch=128



