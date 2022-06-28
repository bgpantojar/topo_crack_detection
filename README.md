# Topo crack detection
This repository contains the codes for crack detection using topological loss function. The methodology hereby implemented was presented in the paper ["TOPO-Loss for continuity-preserving crack detection using deep learning" by Pantoja-Rosero et., al. (2022)](https://doi.org/10.1016/j.conbuildmat.2022.128264)


<p align="center">
  <img src=docs/images/tcd_1.png>
</p>


<p align="center">
  <img src=docs/images/tcd_2.png>
</p>


<p align="center">
  <img src=docs/images/tcd_3.png>
</p>


<p align="center">
  <img src=docs/images/tcd_4.png>
</p>

## How to use it? (Note: tested for ubuntu 18.04lts)

### 1. Clone repository

Clone repository in your local machine. All codes related with method are inside the `src` directory.

### 2. Download data and CNN weights

Example input data can be downloaded from [Dataset for TOPO-Loss for continuity-preserving crack detection using deep learning](https://doi.org/10.5281/zenodo.6769028). This datased contains 3 main folders. data, models results. Extract the three folders and place them inside the repository folder

#### 2a. Repository directory

The repository directory should look as:

```
topo_crack_detection
└───data
└───docs
└───models
└───results
└───src
```

### 3. Environment

Create a conda environment and install python packages. At the terminal in the repository location.

`conda create -n topo_crack_detection python=3.7`

`conda activate topo_crack_detection`

`pip install -r requirements.txt`

`pip3 install torch torchvision`

### 4. Testing method with pusblished models

Open the terminal inside the src folder (with the environment activated -- `conda activate topo_crack_detection`) and write the next command:

`python test.py`

The script by default will call the MSE+TOPO trained model and used it with the full sized images placed inside `data\test_set\images\`. The inference results will be placed inside the folder `results\`.

Note: If want to test another model, change the [path](https://github.com/bgpantojar/topo_crack_detection/blob/fb21c6ebbb8af6017e90173dc82b2251a8f6967c/src/test.py#L30) inside the `test.py` file. If your memory is overflowed, reduce the [patch size](https://github.com/bgpantojar/topo_crack_detection/blob/fb21c6ebbb8af6017e90173dc82b2251a8f6967c/src/test.py#L38)

### 5. Training models

To train the models with the provided data set, ppen the terminal inside the src folder (with the environment activated -- `conda activate topo_crack_detection`) and write the next command:

`python main.py --model_name="your_modeld_choice" --lr=your_learning_rate --n_epoch=your_epoch_number --malis_neg=your_topo_parameter1 --malis_pos=your_topo_parameter1`

The next lines are used to train some of the models presented in the paper.

- MSE model 
`python main.py --model_name="mse" --lr=5e-6 --n_epoch=50`
- TOPO model 
`python main.py --model_name="topo" --lr=3e-5 --n_epoch=50 --malis_neg=100 --malis_pos=10`
- DICE+TOPO model 
`python main.py --model_name="dice+topo" --lr=3e-5 --n_epoch=50 --malis_neg=100 --malis_pos=10`
- MSE+TOPO model
`python main.py --model_name="mse+topo" --lr=3e-5 --n_epoch=50 --malis_neg=100 --malis_pos=10`

Note: malis_neg -> stimulates connectivity. malis_pos -> helps to decrease false-positives. The saved models will be placed inside the `models/` folder.


### 6. Training with your data.

Follow the structure of the `data/` folder and place your data accordingly (training and validation datasets). Note that the ground truth used is derivated from the skeleton of the crack annotation. Train the models following the commands described in 5.

### 7. Testing your models

Once you have trained your modes, change the [model's path](https://github.com/bgpantojar/topo_crack_detection/blob/fb21c6ebbb8af6017e90173dc82b2251a8f6967c/src/test.py#L30) accordingly inside the file `test.py`. Run the command as described in 4.

The results will be saved inside `results` folder. This are formed by predictions as distance maps, thresholded binary image and original images with damage overlayed.

### 8. Citation

We kindly ask you to cite us if you use this project, dataset or article as reference.

Paper:
```
@article{Pantoja-Rosero2020a,
title = {TOPO-Loss for continuity-preserving crack detection using deep learning},
journal = {Construction and Building Materials},
volume = {344},
pages = {128264},
year = {2022},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2022.128264},
url = {},
author = {B.G. Pantoja-Rosero and D. Oner and M. Kozinski and R. Achanta and P. Fua and F. Perez-Cruz and K. Beyer},
}
```
Dataset:
```
@dataset{Pantoja-Rosero2022a-ds,
  author       = {Pantoja-Rosero, Bryan German and
                  Oner, Doruk and
                  Kozinski, Mateusz and
                  Achanta, Radhakrishna and
                  Fua, Pascal and
                  Perez-Cruz, Fernando and
                  Beyer, Katrin},
  title        = {{Dataset for TOPO-Loss for continuity-preserving 
                   crack detection using deep learning}},
  month        = jun,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.0},
  doi          = {10.5281/zenodo.6769028},
  url          = {https://doi.org/10.5281/zenodo.6769028}
}
```
