# SA-SDN

SA-SDN: Symptom Disentanglement Network for Syndrome-Aware Herb Recommendation

## Environment Setup

1. Pytorch 1.12+

2. Python 3.8+
3. torch_geometric 2.1+

## model

- The implementation of SA-SDN(```model.py```);

- The implementation of Attention Mechanism(```attention.py```)

## utils

- Data input and preprocessing(```data_utils.py```)

- Parameter setting(```parser.py```)

## data

You can find the data for testing at the following link: Dataset_XHAd.

## Example to run the codes

```
python train.py --batch_size 50 --numSym 4 --st 0.2
```
