
## Game Theoretic Motion Planner

This repository contains a Python implementation of the Sensitivity-Enhanced Iterated Best Response algorithm for Game Theoretic Motion Planner. 

## Preview

### Blocking Scenario
<img src="media/animation/blocking/blocking_se_ibr_a5.6v1.8c4_with_mpc_a5.6v2.0c4_with_ibr_a5.6v2.2c4_0.gif?raw=true" width="800"/>

### Overtaking Scenario
<img src="media/animation/overtaking/overtaking_se_ibr_a5.6v2.2c4_with_mpc_a5.6v2.0c4_with_ibr_a5.6v1.8c4_0.gif?raw=true" width="800"/>

## References
If you find this project useful in your work, please consider citing following papers:

Game-theoretic planning for self-driving cars in multivehicle competitive scenarios  [[IEEE]](https://ieeexplore.ieee.org/abstract/document/9329208)

```
@article{wang2021game,
  title={Game-theoretic planning for self-driving cars in multivehicle competitive scenarios}, 
  author={Wang, Mingyu and Wang, Zijian and Talbot, John and Gerdes, J Christian and Schwager, Mac},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={4},
  pages={1313--1325},
  year={2021},
  publisher={IEEE}
}
```

A real-time game theoretic planner for autonomous two-player drone racing [[IEEE]](https://ieeexplore.ieee.org/abstract/document/9112709)

```
@article{spica2020real,
  title={A real-time game theoretic planner for autonomous two-player drone racing},
  author={Spica, Riccardo and Cristofalo, Eric and Wang, Zijian and Montijano, Eduardo and Schwager, Mac},
  journal={IEEE Transactions on Robotics},
  volume={36},
  number={5},
  pages={1389--1403},
  year={2020},
  publisher={IEEE}
}
```


## Installation
* We recommend creating a new conda environment:
```
conda env create -f environment.yml
conda activate Game_Theoretic_Planner
```

Run following command in terminal to install the Game_Theoretic_Planner simulator package.
```
python -m pip install -r requirements.txt
pip install -e .
```


## Docs
The following documentation contains documentation and common terminal commands for simulations and testing.

### Generate track
Run
```
python GTP/tests/track_global_position_test.py
```
This allows the generation of the global positions of points on the track center line.

### Performance of single planner
Run
```
python GTP/tests/planner_single_test.py
```
This allows testing the performance of a single MPC planner, IBR planner, or SE-IBR planner. 

### Racing performance of multiple planners
Run
```
python GTP/tests/racing_test.py
```
This allows us to test the racing performance of multiple planners. 

