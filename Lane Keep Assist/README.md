# Lane keep assist
This folder contains:

1. 1st_model : contains the notebook used to train the model, first try
2. 2st_model : contains the notebook used to train the model, second try
3. 3st_model : contains the notebook used to train the model, third try
4. Model Weights: contains the weights of the trained model
5. client side: contains the files required to run the model on jetson nano using cuda

## Usage 

### Pixel summation as control algorithm
#### Without GUI
    `cd client\ side`
    `python3 detectLane.py`
#### With GUI
    `cd client\ side`
    `python3 detectLaneGUI.py`

### Polynomials as control algortihm
#### Without GUI
    `cd client\ side`
    `python3 detectLanePoly.py`
#### With GUI
    `cd client\ side`
    `python3 detectLaneGUIPoly.py`
#### Process on server
    on a laptop or computer
    `cd server\ side`
    `python3 server_lane.py`
    on Jetson Nano
    `cd client\ side`
    `python3 detectLaneClient.py`