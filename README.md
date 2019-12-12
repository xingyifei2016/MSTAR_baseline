# MSTAR_baseline
Baseline for comparison to RotLieNet on the MSTAR dataset. This model serves as a baseline model for [RotLieNet](https://github.com/xingyifei2016/RotLieNet)

## Data Preparation

- First, run `zip -s0 split_data.zip --out data.zip` inside the `split_data` folder.

- Next, extract `data.zip` and set the correct path to the data_polar folder inside the argparse configuration in `baseline.py`


## Getting Started (Training & Testing)


- To train the model: 
```
python baseline.py
```


