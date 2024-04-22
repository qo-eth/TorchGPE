About the package
=================

`TorchGPE` is a Python package designed to solve the Gross-Pitaevskii equaiton and study a dilute Bose-Einstein condensate in arbitrary potentials. 

- It allows to compute the **ground state** wave function of a BEC through imaginary time propagation, as well as **evolving it in real time** under **static and time-dependent potentials**. 
- A :doc:`comprehensive library<fundamentals.potentials>` of potentials is already implemented in TorchGPE. For example, it is possible to study a BEC in an **optical lattice** or one in a **cavity** in just a few lines of code. 
- In addition to the implemented systems, TorchGPE's **modular structure** makes it easy to :doc:`implement new potentials<extending_torchgpe.custom_potentials>` as needed. 
- TorchGPE supports both execution on **CPU** and **GPU** leveraging PyTorch.