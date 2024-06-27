# Deep Learning for Two-Sided Matching

This folder containts the implementation of the paper: [Deep Learning for Two-Sided Matching](https://arxiv.org/pdf/2107.03427)

## Getting Started
The code is written in python3 and requires the following packages
- Numpy
- Numba
- Matplotlib
- PyTorch

## Running the Experiments

We implement the following architectures:
Architecture  | Train filename |
:--------:|:--------------|
MLP | train_MLP.py |
CNN  | train_CNN.py |

To run MLP, do
```
python <train_filename> -n <num_agents> -p <truncation_probability> -c <correlation_probability> -l <lambda>
```

To change other hyperparameters, visit the corresponding file and modify the ```Args``` class.  
The logfiles and the saved models can be found in ```experiments/``` folder

## Citing the Project

Please cite our work if you find our code/paper is useful to your work.
```
@article{ravindranath2021deep,
  title={Deep learning for two-sided matching},
  author={Ravindranath, Sai Srivatsa and Feng, Zhe and Li, Shira and Ma, Jonathan and Kominers, Scott D and Parkes, David C},
  journal={arXiv preprint arXiv:2107.03427},
  year={2021}
}
```
