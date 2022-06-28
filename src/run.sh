#!/bin/bash -l

python main.py --model_name="mse" --lr=5e-6 --n_epoch=50 
python main.py --model_name="topo" --lr=3e-5 --n_epoch=50 --malis_neg=100 --malis_pos=10
python main.py --model_name="dice+topo" --lr=3e-5 --n_epoch=50 --malis_neg=100 --malis_pos=10
python main.py --model_name="mse+topo" --lr=3e-5 --n_epoch=50 --malis_neg=100 --malis_pos=10
