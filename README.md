# Overview
23th place solution to the competition NeurIPS [Auto-Bidding in Uncertain Environment] 2024.
https://tianchi.aliyun.com/home/science/scienceDetail?spm=a2c22.12281920.0.0.35502e0aQJqsUz&userId=1095280667830
```
https://tianchi.aliyun.com/competition/entrance/532226
```
Based on offline reinforcement algorithms. 

## Dependencies
```
conda create -n nips-bidding-env python=3.9.12 pip=23.0.1
conda activate nips-bidding-env
pip install -r requirements.txt
```

# Usage
```
Follow the instructions and project structure given:
https://github.com/alimama-tech/NeurIPS_Auto_Bidding_General_Track_Baseline
```

## strategy training
### reinforcement learning-based bidding


#### BPPO Model
Load the training data and train the BPPO bidding strategy.
```
python main/main_bppo.py 
```

#### CQL Model
Load the training data and train the CQL bidding strategy.
```
python main/main_cql.py 
```
## offline evaluation
Load the training data to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```
