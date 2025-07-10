# M3FED
# Cross-Modal Meta Consensus for Heterogeneous Federated Learning [[PDF Download]()]

<div align="center">
  <img src="./Fig/framework.png" alt="图片名称1" style="width: 80%; display: inline-block; margin-right: 5%;" />
</div>
<p align="center"><em>The network architecture of the proposed framework. Client side: Localized Training Via Dual-level Optimization. Server-side: Cross Modal Meta Aggregation. 𝑓𝜃 is the meta learner, and 𝐺 is the shared consensus operator.</em></p>

## Installation Guide

### Environment
 We manage environments with Conda. To set up the environment, follow these steps:
```
pip install -r requirements.txt
```


## Preprocess dataset

The dataset for this project utilizes a Dirichlet parameter-based partitioning method to allocate client data. The data feature partitioning of the dataset are implemented in the /data/data_preprocess.py file. Before the project runs, feature extraction processing needs to be performed on the data in the dataset.

## Run the experiment
Before run the experiment, please make sure that the dataset is downloaded and preprocessed already.

```
python main.py
```

## References

Thanks for the creative ideas of the pioneer researches:

- https://github.com/KarhouTam/Per-FedAvg

## Citing our work

The preprint can be cited as follows

```bibtex
@inproceedings{li2024cross,
  title={Cross-Modal Meta Consensus for Heterogeneous Federated Learning},
  author={Li, Shuai and Qi, Fan and Zhang, Zixin and Xu, Changsheng},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={975--984},
  year={2024}
}
```
