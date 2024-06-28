# Auto Parking 

## Overview
This is auto parking use case, the folder consists of 3 folders simulation, server-side and client-side.

The control algorithm follows the following flow
![Alt text](/assets/model.png)

## structure
1. Simulation
2. Server-side
    This runs on computer ex. laptop
3. client-side
    This runs on Jetson Nano

## Note
as github refuses to upload files larger than 100MB, please download the weights from the following link, and put it on Auto Parking folder:
[Weight](https://drive.google.com/file/d/1pmd_C4H4LU6yaaCp-9oc2e_vJ06283X_/view?usp=sharing)

## Usage

make sure LiDAR and SLAM are running

#### Running the Simulation
To run the simulation, navigate to the `simulation` directory and execute `top_module.py` using Python 3:
`cd Simulation`
`python3 top_module.py`

#### Running server
`cd Server\ Side`
`python3 box_finder_server.py`

#### Running client
`cd Client\ Side`
`python3 topclient `
