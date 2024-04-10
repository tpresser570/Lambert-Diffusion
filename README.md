# Diffusion Models for Generating Ballistic Spacecraft Transfers

This repo is forked from the official implementation of NCSNV2 from the paper [Improved Techniques for Training Score-Based Generative Models](http://arxiv.org/abs/2006.09011). 
The instructions are the same as those in the original implementation, with small tweaks to work with trajectory data. 

by [Tyler Presser](http://tpresser570.github.io/), USC. 
-----------------------------------------------------------------------------------------

## Running Experiments

### Earth-Mars Transfer Data
Use the Google Drive [LINK](https://drive.google.com/drive/folders/1jAYGXROnBbnWZAR7X-bephRyaON6s3xk?usp=sharing) to access the dataset used to compute the results shown in the paper. Download and extract the zip file. Ensure the path in your config file points to the data you want to use. The trajectory files, which are saved as pickle binary files, are automatically converted to images for the model to process using `datasets/unconditional_lambert.py`, which reads all the pickle files from the data folder and imports them to a pytorch datloader, which the model will access. Your config file should specify what data you want to use and where it is from. 

The dataset is structured as:

```bash
<extended_mars_transfers> # a folder containting all the data
├── 8 # all trajectories of size 8
├── 16 # all trajectories of size 16
│   └──
│      ├── csv #
|           └── 
|                |- sample_inital_conditions.csv # a dataframe of all inital conditions used to generate lambert solutions
|                |- traj_names_df.csv # a dataframe of all trajectory .pkl file names
|                |- traj_names_scaled_df.csv # a dataframe of all trajectory .pkl file names for scaled trajectories [0,1]
│      ├── fea #
|           └── 
|                |- initial_conditions_mars_2d_km.fea # feather table of all inital conditions 
│      ├── npy #
|           └── 
|                |- initial_conditions_mars_2d_km.npy # npy array of all inital conditions in km
|                |- initial_conditions_mars_2d.npy # npy array of all intial conditions in m
│      ├── pkl # a folder of all original trajectory pkl files
│      ├── scaled_pkl # a folder of all original trajectory pkl files
│      ├── ranges #
|           └── 
|                |- ranges.csv # a csv containing all the ranges used to scale the trajectories [0,1]
├── 64 # all trajectories of size 64
├── 1024 # all trajectories of size 1024


```



### Project structure

`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [--test] [--sample]
               [--fast_fid] [--resume_training] [-i IMAGE_FOLDER] [--ni]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name
                        of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --sample              Whether to produce samples from the model
  --fast_fid            Whether to do fast fid test
  --resume_training     Whether to resume training
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  --ni                  No interaction. Suitable for Slurm Job launcher
```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      ├── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
│      ├── config.yml # the configuration file for training this model
│      ├── stdout.txt # all outputs to the console during training
│      └── samples # all samples produced during training
├── fid_samples # contains all samples generated for fast fid computation
│   └── <i> # a folder named by the argument `-i` specified to main.py
│      └── ckpt_x # a folder of image samples generated from checkpoint_x.pth
├── image_samples # contains generated samples
│   └── <i>
│       └── image_grid_x.png # samples generated from checkpoint_x.pth       
└── tensorboard # tensorboard files for monitoring training
    └── <doc> # this is the log_dir of tensorboard
```

### Training

For example, we can train an NCSNv2 on the Earth-Mars transfer data by running the following.

```bash
python main.py --config unconditional_lambert.yml --doc lambert
```

Log files will be saved in `<exp>/logs/lambert`.

### Sampling

If we want to sample from NCSNv2 on the Earth-Mars transfer data, we can edit `unconditional_lambert.yml` to specify the `ckpt_id` under the group `sampling`, and then run the following

```bash
python main.py --sample --config unconditional_lambert.yml -i lambert_samples
```

Samples will be saved in `<exp>/image_samples/lambert_samples`.

### Evaluating Trajectories 

We can evaluate trajectories using both methods proposed in the paper by utlizing the `Trajectory_Eval_Scripts/` that are separated into Python and Julia. Note that after sampling the model outputs will be saved as `.pth` files. Use the scripts in the `Conversions/` folder to convert these files to numpy arrays that can be used for processing and evaluation. To convert samples you will also need the `ranges.csv` file from the dataset to convert from pixels to states.

## References

If you find the code/idea useful for your research, please consider citing the original work by Song and Ermon:

```bib
@inproceedings{song2020improved,
  author    = {Yang Song and Stefano Ermon},
  editor    = {Hugo Larochelle and
               Marc'Aurelio Ranzato and
               Raia Hadsell and
               Maria{-}Florina Balcan and
               Hsuan{-}Tien Lin},
  title     = {Improved Techniques for Training Score-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
               on Neural Information Processing Systems 2020, NeurIPS 2020, December
               6-12, 2020, virtual},
  year      = {2020}
}
```


