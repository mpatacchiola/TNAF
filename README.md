# TNAF
Official repository of the paper "Transformer Neural Autoregressive Flows", ICML 2024 (SPIG Workshop) [[arXiv]](https://arxiv.org/abs/2401.01855)

Installation
------------

**Install the prerequisites**

- torch==2.0.1
- torchvision==0.15.2

You can install the environment via conda as follows:

```
conda env create -f env.yml
conda activate torch2env
```

**Download datasets**

```
wget https://zenodo.org/record/1161203/files/data.tar.gz
tar -zxvf data.tar.gz
rm -rf data/mnist/
rm -rf data/cifar10/
rm data.tar.gz
mkdir checkpoints
```

Usage
-----

Refer to the paper for the exact configuration of the model.

Example of command for training:

```
python train.py --dataset="miniboone" --method="tnaf" --device="cuda:0" --seed=1
```

The best model (on validation score) is saved in the checkpoints folder.
At thest time the best model is automatically loaded and evaluated on the test set:

```
python test.py --dataset="miniboone" --method="tnaf" --device="cuda:0"
```

Citation
-------

```
Patacchiola, M., Shysheya, A., Hofmann, K., & Turner, R. E. (2024). Transformer neural autoregressive flows. arXiv preprint arXiv:2401.01855.
```

**BibTeX**

```
@article{patacchiola2024transformer,
  title={Transformer neural autoregressive flows},
  author={Patacchiola, Massimiliano and Shysheya, Aliaksandra and Hofmann, Katja and Turner, Richard E},
  journal={arXiv preprint arXiv:2401.01855},
  year={2024}
}
```

Related work
------------

Block Neural Autoregressive Flow (BNAF) [[GitHub]](https://github.com/nicola-decao/BNAF) [[arXiv]](http://arxiv.org/abs/1904.04676)
