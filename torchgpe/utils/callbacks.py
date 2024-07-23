import torch

from abc import ABCMeta


class Callback(metaclass=ABCMeta):
    """Base class for callbacks.

    Before a simulation starts, it is provided with the instance of the :class:`gpe.bec2D.gas.Gas` (stored in the :py:attr:`gpe.utils.callbacks.Callback.gas` variable) and with a dictionary of parameters for the simulation (stored in :py:attr:`gpe.utils.callbacks.Callback.propagation_params`)
    """

    def __init__(self) -> None:
        #: gpe.bec2D.gas.Gas: The instance of the :class:`gpe.bec2D.gas.Gas` class. Populated when the simulation starts.
        self.gas = None
        #: dict: A dictionary of parameters for the simulation. Populated when the simulation starts.
        self.propagation_params = None

    def set_gas(self, gas):
        self.gas = gas

    def set_propagation_params(self, propagation_params):
        self.propagation_params = propagation_params
        self.early_stop = False

    def on_propagation_begin(self):
        """Function called by the :class:`gpe.bec2D.gas.Gas` class before the simulation begins
        """
        pass

    def on_propagation_end(self):
        """Function called by the :class:`gpe.bec2D.gas.Gas` class after the simulation ends
        """
        pass

    def on_epoch_begin(self, epoch: int):
        """Function called by the :class:`gpe.bec2D.gas.Gas` at the beginning of each epoch

        Args:
            epoch (int): The epoch number
        """
        pass

    def on_epoch_end(self, epoch: int):
        """Function called by the :class:`gpe.bec2D.gas.Gas` at the end of each epoch

        If the callback sets the :py:attr:`gpe.utils.callbacks.Callback.early_stop` variable to ``True``, the simulation will stop right after this function is called.
        
        Args:
            epoch (int): The epoch number
        """
        pass


class LInfNorm(Callback):
    """Callback computing the :math:`L_\\infty` norm of the wavefunction

    The :math:`L_\\infty` norm is defined as:

    .. math::

        L_\\infty = \\text{max}_{(x,y)}|\\Psi_t - \\Psi_{t+\\Delta t}|

    Args:
        compute_every (int): Optional. The number of epochs after which the norm is computed. Defaults to 1.
        print_every (int): Optional. The number of epochs after which, if computed, the norm is also printed. Defaults to 1.
    """

    def __init__(self, compute_every=1, print_every=1) -> None:

        super().__init__()
        #: list: A list of the computed norms
        self.norms = []
        self.compute_every = compute_every
        self.print_every = print_every

    def on_epoch_begin(self, epoch: int):
        """At the beginning of an epoch, if its number is a multiple of ``compute_every`` stores the wave function of the gas

        Args:
            epoch (int): The epoch number
        """
        if epoch % self.compute_every != 0:
            return

        self.psi = self.gas.psi

    def on_epoch_end(self, epoch: int):
        """At the end of an epoch, if its number is a multiple of ``compute_every`` uses the stored wave function of the gas to compute 
        the :math:`L_\\infty` norm. If the epoch number is a multiple of ``print_every`` as well, the value of the norm is printed on screen.

        Args:
            epoch (int): The epoch number
        """

        if epoch % self.compute_every != 0:
            return

        psi = self.gas.psi

        self.norms.append(torch.max(torch.abs(psi-self.psi)).cpu())
        del self.psi

        if epoch % self.print_every == 0:
            print(self.norms[-1])


class L1Norm(Callback):
    """Callback computing the :math:`L_1` norm of the wavefunction

    The :math:`L_1` norm is defined as:

    .. math::

        L_1 = \\sum_{(x,y)}|\\Psi_t - \\Psi_{t+\\Delta t}| \\, dx \\, dy

    Args:
        compute_every (int): Optional. The number of epochs after which the norm is computed. Defaults to 1.
        print_every (int): Optional. The number of epochs after which, if computed, the norm is also printed. Defaults to 1.
    """

    def __init__(self, compute_every=1, print_every=1) -> None:
        super().__init__()
        #: list: A list of the computed norms
        self.norms = []
        self.compute_every = compute_every
        self.print_every = print_every

    def on_epoch_begin(self, epoch):
        """At the beginning of an epoch, if its number is a multiple of ``compute_every`` stores the wave function of the gas

        Args:
            epoch (int): The epoch number
        """

        if epoch % self.compute_every != 0:
            return

        self.psi = self.gas.psi

    def on_epoch_end(self, epoch):
        """At the end of an epoch, if its number is a multiple of ``compute_every`` uses the stored wave function of the gas to compute 
        the :math:`L_1` norm. If the epoch number is a multiple of ``print_every`` as well, the value of the norm is printed on screen.

        Args:
            epoch (int): The epoch number
        """

        if epoch % self.compute_every != 0:
            return

        psi = self.gas.psi

        self.norms.append((torch.sum(torch.abs(psi-self.psi))
                          * self.gas.dx*self.gas.dy).cpu())
        del self.psi

        if epoch % self.print_every == 0:
            print(self.norms[-1])


class L2Norm(Callback):
    """Callback computing the :math:`L_2` norm of the wavefunction

    The :math:`L_2` norm is defined as:

    .. math::

        L_2 = \\sqrt{\\sum_{(x,y)}|\\Psi_t - \\Psi_{t+\\Delta t}|^2 \\, dx \\, dy}

    Args:
        compute_every (int): Optional. The number of epochs after which the norm is computed. Defaults to 1.
        print_every (int): Optional. The number of epochs after which, if computed, the norm is also printed. Defaults to 1.
    """

    def __init__(self, compute_every=1, print_every=1) -> None:
        super().__init__()
        #: list: A list of the computed norms
        self.norms = []
        self.compute_every = compute_every
        self.print_every = print_every

    def on_epoch_begin(self, epoch):
        """At the beginning of an epoch, if its number is a multiple of ``compute_every`` stores the wave function of the gas

        Args:
            epoch (int): The epoch number
        """

        if epoch % self.compute_every != 0:
            return

        self.psi = self.gas.psi

    def on_epoch_end(self, epoch):
        """At the end of an epoch, if its number is a multiple of ``compute_every`` uses the stored wave function of the gas to compute 
        the :math:`L_2` norm. If the epoch number is a multiple of ``print_every`` as well, the value of the norm is printed on screen.

        Args:
            epoch (int): The epoch number
        """
        if epoch % self.compute_every != 0:
            return

        psi = self.gas.psi

        self.norms.append(torch.sqrt(
            torch.sum(torch.abs(psi-self.psi)**2)*self.gas.dx*self.gas.dy).cpu())
        del self.psi

        if epoch % self.print_every == 0:
            print(self.norms[-1])
