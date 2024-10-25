# Overview
23 place solution to the competition NeurIPS [Auto-Bidding in Uncertain Environment] 2024.


Based on offline reinforcement algorithms. 

. 


## Dependencies
```
conda create -n nips-bidding-env python=3.9.12 pip=23.0.1
conda activate nips-bidding-env
pip install -r requirements.txt
```

# Usage

Follow the instructions and project structure given:
https://github.com/cerenov94/nips-bidding-comp/blob/main/bidding_train_env/bppo/replay_buffer_upd.py
```

## strategy training
### reinforcement learning-based bidding

#### BPPO Model
Load the training data and train the IQL bidding strategy.
```
python main/main_bppo.py 
```
```
```
#### CQL Model
Load the training data and train the BC bidding strategy.
```
python main/main_cql.py 
```

## offline evaluation
Load the training data to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```



   
