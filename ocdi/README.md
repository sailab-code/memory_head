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
The code repository is composed by the following files and folders:

    mh :                 folder containing the source code of our mh model
    competitors :                   source folder for the implementation of continual learning baselines and competitors  
    d2d :                   folder containing 2D dataset (MODES in the paper) generators and utils
        check_2d_datasets.py:   file for checking the shapes of the built 2d datasets
        generate_2d_datasets.py:   file for the customizable creation of the 2D datasets
        utils_2d_datasets.py : code utils for handling the 2D datasets 
        example_2d_vanilla.py: example script for a vanilla model
        example_2d_mh.py: example script for the usage and definition of MHs

    datasets :              folder containing the non-stationary imagenet dataset files    
    main.py :             experiments runner


USAGE EXAMPLE
-------------
Have a look at the example script available in the path `d2d/example_2d_mh.py` for a practical example on MHs definition and usage!


DATASETS GENERATION
-------------------

The datasets needed for the experimental campaign can be generated with the provided code, that is described in the
following.

### 2D datasets

We share the code to generate the MODES dataset described in the main paper in
the `d2d/generate_2d_datasets.py` file. The user can decide the number of samples, the class sample ratio, the
distributions shape and centers. The script handles the creation of the CL setting described in the paper (*CDI*).
The script must be runned in order to create the data needed for running the experiments. 
The `d2d/check_2d_datasets.py` script helps in better visualizing the created data.   

### Non-stationary Imagenet

We followed the data setting proposed in "Online continual learning in image classification: An empirical
survey" [Mai et al., 2022] using a subset of 100 class categories, following three distributions (original, small amount
of noise, large amount of noise). Data can be downloaded
from [KAGGLE](https://www.kaggle.com/account/login?titleType=dataset-downloads&showDatasetDownloadSkip=False&messageId=datasetsWelcome&returnUrl=%2Fdatasets%2Fwhitemoon%2Fminiimagenet%3Fresource%3Ddownload)
and placed in the folder:

```
datasets/mini_imagenet/
```

Perturbed versions of the dataset are going to be generated automatically and cached on disk at the first run or
the `main.py` script (see below for usage details).


RUNNING EXPERIMENTS
-------------------

We tested our code with `PyTorch 1.10`. Please install the other required dependencies by running:

```
pip install -r requirements.txt
```

We provide a `main.py`script to easily test the proposed model. The PyTorch device is chosen through the `--device`
argument (`cpu`, `cuda:0`,
`cuda:1`, etc.).

    usage: main.py [-h] [--model {lp,mlp,mh}] [--bias BIAS] [--watch WATCH] [--simple SIMPLE] [--hidden HIDDEN]
               [--benchmark {bi-modals_IID,bi-modals_CI,bi-modals_CDI,bi-modals_CDID,bi-moons_IID,bi-moons_CI,bi-moons_CDI,bi-moons_CDIDimagenet}] [--beta_m BETA_M] [--beta_k BETA_K] [--weight_decay WEIGHT_DECAY]
               [--optimizer OPTIMIZER] [--loss {xent,hinge}] [--wandb WANDB] [--key_mem_units KEY_MEM_UNITS] [--delta DELTA] [--psi {identity,sign}] [--key_size KEY_SIZE] [--gamma_alpha GAMMA_ALPHA] [--tau_alpha TAU_ALPHA]
               [--tau_mu TAU_MU] [--tau_eta TAU_ETA] [--upd_m {vanilla,WTA}] [--upd_k {ad_hoc_WTA,grad_WTA,grad_not_WTA}] [--scramble SCRAMBLE] [--shared_keys SHARED_KEYS] [--draw_plots DRAW_PLOTS] [--seed SEED] [--device DEVICE]
               [--competitor COMPETITOR] [--buffer_size BUFFER_SIZE] [--buffer_batch_size BUFFER_BATCH_SIZE] [--gdumb_epochs GDUMB_EPOCHS] [--ensembled_models ENSEMBLED_MODELS] [--eval_chunk_size EVAL_CHUNK_SIZE]

Argument description/mapping with respect to the paper notation:

      --model  : the neural architecture to be used
      --bias BIAS : boolean Flag for the use of bias weights
      --watch WATCH : wand watch model
      --hidden HIDDEN : amount of neurons in the hidden layer (for mlp only )
      --benchmark  : the dataset to be used; the 2D datasets are referred to as bi-modals (MODES in the paper), and bi-moons (MOONS in the paper)
      --beta_m BETA_M : \ro in the paper (step size)
      --beta_k BETA_K : \beta in the paper (key update strenght)
      --weight_decay WEIGHT_DECAY : weight decay to be used  
      --optimizer OPTIMIZER : which optimizer to be used
      --loss :  which loss to be use
      --wandb WANDB : flag to use wandb for logging
      --key_mem_units KEY_MEM_UNITS : number of memory units
      --delta DELTA : \delta in the paper (top-delta in attention)
      --psi   : type of psi function 
      --key_size KEY_SIZE : key dimension when using a reduction form of psi
      --gamma_alpha GAMMA_ALPHA :  same in the paper, softmax temperature 
      --tau_alpha TAU_ALPHA : same in the paper, for scrambling 
      --tau_mu TAU_MU : same in the paper, for scrambling
      --tau_eta TAU_ETA : same in the paper, for scrambling
      --upd_m {vanilla,WTA} : memory update mode  (vanilla == a_M in paper ablations, WTA == proposed in paper ablations)
      --upd_k {ad_hoc_WTA,grad_WTA} : key update mode (grad_WTA == g_K in paper ablations, ad_hoc_WTA == proposed in paper ablations                                                                    )
      --scramble SCRAMBLE : flag to activate scrambling
      --shared_keys SHARED_KEYS : use shared keys in all the layer' neurons
      --draw_plots DRAW_PLOTS : drawing 2d separation surfaces 
      --seed SEED  : seed for the run
      --device DEVICE 
      --competitor COMPETITOR : to be specified when using other CL competitors
      --buffer_size BUFFER_SIZE : buffer size for competitors only
      --buffer_batch_size BUFFER_BATCH_SIZE : buffer batch size 
      --gdumb_epochs GDUMB_EPOCHS : number of epochs for GDumb
      --ensembled_models ENSEMBLED_MODELS : number of ensembled models (the model architecture is repeated)
      --eval_chunk_size EVAL_CHUNK_SIZE : utility for imagenet evaluation

### Example command lines

The best selected hyper-parameters can be found in the Appendix of the paper. In the following, some example command lines
to run the experiments:

    # running a 2D dataset experiment with a mh based model
    python main.py --benchmark=bi-modals_CDI --beta_k=0.001 --beta_m=0.01 --bias=true --delta=2 --device=cpu --draw_plots=false --gamma_alpha=5 --hidden=5 --key_mem_units=8  --model=mh  --scramble=true --seed=1234 --shared_keys=true --tau_alpha=0.95 --tau_eta=50 --tau_mu=50 --upd_k=ad_hoc_WTA --upd_m=WTA --weight_decay=0

    # running a CL competitor on the MODES-CDI dataset
    python main.py  --benchmark=bi-modals_CDI --beta_m=0.001 --bias=true --buffer_batch_size=1 --buffer_size=8 --competitor=MIR --device=cpu --hidden=25 --model=mlp --seed=12345678 

    # running mh on the ns-imagenet dataset
    python main.py  --benchmark=imagenet --beta_k=0.001 --beta_m=0.0001 --bias=true --delta=2 --device=cuda  --gamma_alpha=5 --hidden=50 --key_mem_units=100 --loss=xent --model=mh --scramble=true --seed=1234 --shared_keys=true --tau_alpha=0.7 --tau_eta=500 --tau_mu=50 --upd_k=ad_hoc_WTA --upd_m=WTA --weight_decay=0.001

COLLECTING THE FINAL METRICS
----------------------------

The final metrics are printed to screen at the end of each run.




