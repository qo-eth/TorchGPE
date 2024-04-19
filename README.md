# TorchGPE

## Description

TorchGPE is a python package for solving the Gross-Pitaevskii equation (GPE). 
The numerical solver is designed to integrate a wave function in a variety of linear and non-linear potentials. 
The code is based on symmetric Fourier split-step propagation, both in real and imaginary time.
It has a modular approach that allows the integration of arbitrary self-consistent and time-dependent potentials.

## Installation

The code uses PyTorch tensors to speed up the calculation (which heavily relies on FFTs). You can check if cuda is available on your system with

```python
import torch
torch.cuda.is_available()
```
If not, the default device is set to CPU. You can choose to run on CPU in any case, even if Cuda is available. 

### Installing via pip

The package can be installed via pip 

```shell
pip install git+https://github.com/qo-eth/TorchGPE.git
```

Or, if you already downloaded the repository, run

```shell
pip install .
```

## Documentation


## Credits

The code was created within the [Quantum Optics](https://www.quantumoptics.ethz.ch/) research group at ETH Zurich
