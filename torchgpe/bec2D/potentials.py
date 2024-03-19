from __future__ import annotations
from typing import Union, Callable

import scipy.constants as spconsts
import numpy as np
import torch

from ..utils.potentials import LinearPotential, NonLinearPotential, any_time_dependent_variable, time_dependent_variable


# --- Linear potentials ---
class Zero(LinearPotential):
    """Zero potential. It is equivalent to not applying any potential at all.
    """

    def __init__(self):
        super().__init__()

    def get_potential(self, X: torch.tensor, Y: torch.tensor, time: float = None):
        return torch.zeros_like(X)


class Trap(LinearPotential):
    """Harmonic trapping potential

    Args:
        omegax (Union[float, Callable]): The frequency along the x axis of the harmonic oscillator. It can be set to be either a constant or a function of time.
        omegay (Union[float, Callable]): The frequency along the y axis of the harmonic oscillator. It can be set to be either a constant or a function of time.
    """

    def __init__(self, omegax: Union[float, Callable], omegay: Union[float, Callable]):
        super().__init__()

        self.omegax = omegax
        self.omegay = omegay

    def on_propagation_begin(self):
        self.is_time_dependent = any_time_dependent_variable(
            self.omegax, self.omegay)

        self._omegax = time_dependent_variable(self.omegax)
        self._omegay = time_dependent_variable(self.omegay)

    def get_potential(self, X: torch.tensor, Y: torch.tensor, time: float = None):
        return 2*(np.pi/self.gas.adim_pulse)**2*((self._omegax(time)*X)**2+(self._omegay(time)*Y)**2)


class Lattice(LinearPotential):
    """Lattice potential

    Args:
        V0 (Union[float, Callable]): The lattice depth in units of the recoil energy. It can be set to be either a constant or a function of time.
        lam (float): The wave length of the lattice.
        theta (float): The angle of the lattice in the 2D plane.
        phi (Union[float, Callable]): The phase of the lattice.
    """

    def __init__(self, V0: Union[float, Callable] = 0, lam: float = 1e-6, theta: float = 0, phi: Union[float, Callable] = 0, w0: float = np.inf):
        super().__init__()

        self.V0 = V0
        self.lam = lam
        self.theta = theta
        self.phi = phi
        self.w0 = w0

    def on_propagation_begin(self):
        self._lam = self.lam/self.gas.adim_length
        self._k = 2*np.pi/self._lam
        self._w0 = self.w0/self.gas.adim_length
        self._rayleigh = np.pi*self._w0**2/self._lam
        self.Er = 0.5 * (spconsts.hbar*self._k /
                         self.gas.adim_length)**2 / self.gas.mass

        self.is_time_dependent = any_time_dependent_variable(
            self.V0, self.phi)
        self._V0 = time_dependent_variable(self.V0)
        self._phi = time_dependent_variable(self.phi)

        self._w = self._w0 * \
            torch.sqrt(1+(self.gas.X*np.cos(self.theta) +
                       self.gas.Y*np.sin(self.theta))**2/self._rayleigh**2)
        self._R = self.gas.X*np.cos(self.theta) + self.gas.Y*np.sin(self.theta) + self._rayleigh**2/(self.gas.X*np.cos(self.theta) +
             self.gas.Y*np.sin(self.theta))
        self._gouy = torch.atan(
            (self.gas.X*np.cos(self.theta) + self.gas.Y*np.sin(self.theta))/self._rayleigh)
        self._Epos = lambda time: (1/torch.sqrt(1+(self.gas.X*np.cos(self.theta) + self.gas.Y*np.sin(self.theta))**2/self._rayleigh**2) * torch.exp(-(-self.gas.X*np.sin(self.theta) + self.gas.Y*np.cos(self.theta))**2/self._w**2) *
                                   (
            torch.exp(1j *
                                  (self._k*((self.gas.X*np.cos(self.theta) + self.gas.Y*np.sin(self.theta)) + (-self.gas.X*np.sin(self.theta) + self.gas.Y*np.cos(self.theta))**2/(2*self._R)) - self._gouy
                                   ))
            +
            torch.exp(-1j *
                      (self._k*((self.gas.X*np.cos(self.theta) + self.gas.Y*np.sin(self.theta)) + (-self.gas.X*np.sin(self.theta) + self.gas.Y*np.cos(self.theta))**2/(2*self._R)) - self._gouy + self._phi(time)
                       ))
        )
        )

    def get_potential(self, X: torch.tensor, Y: torch.tensor, time: float = None):
        return self._V0(time) * self.Er/(spconsts.hbar*self.gas.adim_pulse) * torch.abs(self._Epos(time))**2/4


class SquareBox(LinearPotential):
    """Square box potential

    Args:
        V (float): The depth of the box.
        D (float): The size of the box.
    """

    def __init__(self, V: float, D: float):
        super().__init__()
        self.V = V
        self.D = D

    def on_propagation_begin(self):
        self._V = self.V/(spconsts.hbar*self.gas.adim_pulse)
        self._D = self.D/self.gas.adim_length
        self._box = self._V * (1-torch.heaviside(-torch.abs(self.gas.X)+self._D/2, torch.ones_like(self.gas.X)) *
                               torch.heaviside(-torch.abs(self.gas.Y)+self._D/2, torch.ones_like(self.gas.Y)))

    def get_potential(self, X: torch.tensor, Y: torch.tensor, time: float = None):
        return self._box


class RoundBox(LinearPotential):
    """Round box potential

    Args:
        V (float): The depth of the box.
        D (float): The diameter of the box.
    """

    def __init__(self, V: float, D: float):
        super().__init__()
        self.V = V
        self.D = D

    def on_propagation_begin(self):
        self._V = self.V/(spconsts.hbar*self.gas.adim_pulse)
        self._D = self.D/self.gas.adim_length
        self._box = self._V*(1-torch.heaviside((self._D/2)**2 -
                             self.gas.X**2-self.gas.Y**2, torch.ones_like(self.gas.X)))

    def get_potential(self, X: torch.tensor, Y: torch.tensor, time: float = None):
        return self._box


# --- Non linear potentials ---
class Contact(NonLinearPotential):
    """Contact interactions potential

    Args:
        a_s (float): The scattering length in units of the Bohr radius.
        a_orth (float): The renormalization parameter for the scattering length to account for the missing third dimension.
    """

    def __init__(self, a_s: float = 100, a_orth: float = 1e-6):
        super().__init__()

        self.a_s = a_s
        self.a_orth = a_orth

    def on_propagation_begin(self):
        self._a_s = self.a_s*spconsts.codata.value("Bohr radius")
        self._g = np.sqrt(8*np.pi)*self.gas.N_particles*self._a_s/self.a_orth

    def potential_function(self, X: torch.tensor, Y: torch.tensor, psi: torch.tensor, time: float = None):
        return self._g*torch.abs(psi)**2


class DispersiveCavity(NonLinearPotential):
    """Transversally pumped dispersive cavity potential

    Args:
        lattice_depth (Union[float, Callable]): The lattice depth in units of the recoil energy. It can be set to be either a constant or a function of time.
        atomic_detuning (float): The atomic frequency detuning with respect to the pump.
        cavity_detuning (Union[float, Callable]): The cavity's frequency detuning with respect to the pump. It can be set to be either a constant or a function of time.
        cavity_decay (float): The cavity's decay rate.
        cavity_coupling (float): The coupling constant between the gas and the cavity.
        cavity_angle (float, optional): The angle in the 2D plane of the cavity. Defaults to :math:`0`
        pump_angle (float, optional): The angle in the 2D plane of the transversal pump. Defaults to :math:`\\pi/3`
        waist (float, optional): the waist of the gaussian beam. Defaults to infinity
    """

    def __init__(self, lattice_depth: Union[float, Callable], atomic_detuning: float, cavity_detuning: Union[float, Callable], cavity_decay: float, cavity_coupling: float, cavity_angle: float = 0, pump_angle: float = np.pi/3, waist: float = np.inf):

        super().__init__()

        self.lattice_depth = lattice_depth
        self.atomic_detuning = atomic_detuning
        self.cavity_detuning = cavity_detuning
        self.cavity_decay = cavity_decay
        self.cavity_coupling = cavity_coupling
        self.cavity_angle = cavity_angle
        self.pump_angle = pump_angle
        self.waist = waist

    def on_propagation_begin(self):
        self.is_time_dependent = any_time_dependent_variable(
            self.cavity_detuning, self.lattice_depth)

        self._cavity_detuning = time_dependent_variable(self.cavity_detuning)
        self._lattice_depth = time_dependent_variable(self.lattice_depth)

        self.g0 = 2*np.pi*self.cavity_coupling
        self._atomic_detuning = 2*np.pi*self.atomic_detuning
        self.kappa = 2*np.pi*self.cavity_decay
        self.freq_d2 = self.gas.d2_pulse
        self.lambda_pump = 2*np.pi*spconsts.c / \
            (self.freq_d2+self._atomic_detuning)
        self.adim_lambda_pump = self.lambda_pump/self.gas.adim_length
        self.k_pump = 2*np.pi/self.lambda_pump
        self.adim_k_pump = 2*np.pi/self.adim_lambda_pump
        self.Er = 0.5 * (spconsts.hbar*self.k_pump)**2 / self.gas.mass
        self.U0 = self.g0**2 / self._atomic_detuning
        self._adim_waist = self.waist / self.gas.adim_length

        R_pump = self.gas.X * \
            np.cos(self.pump_angle) + self.gas.Y * np.sin(self.pump_angle)
        R_pump_orth = - self.gas.X * \
            np.sin(self.pump_angle) + self.gas.Y * np.cos(self.pump_angle)
        R_cavity = self.gas.X * \
            np.cos(self.cavity_angle) + self.gas.Y * np.sin(self.cavity_angle)
        self.COS2 = torch.cos(self.adim_k_pump*R_cavity)**2
        self.COS = torch.cos(self.adim_k_pump*R_pump) * \
            torch.cos(self.adim_k_pump*R_cavity)
        self.c1 = self.gas.N_particles*self.gas.dx*self.gas.dy
        self.c3 = self.c1*self.U0
        self.eta_prefactor = np.sqrt(self.Er*np.abs(self._atomic_detuning)/spconsts.hbar) * \
            self.g0/self._atomic_detuning
        self._gaussian_profile = 1/(1+(self.adim_lambda_pump*R_pump/(np.pi*self._adim_waist**2))**2)*torch.exp(
            -2 * R_pump_orth**2/(self._adim_waist**2 + (self.adim_lambda_pump*R_pump/(np.pi*self._adim_waist))**2))
        self._pump_lattice = np.sign(self._atomic_detuning) * self.Er * torch.cos(
            self.adim_k_pump*R_pump)**2 / (spconsts.hbar * self.gas.adim_pulse) * self._gaussian_profile
        self._cavity_lattice = self.COS2 * self.U0 / self.gas.adim_pulse

    def get_alpha(self, psi: torch.tensor, time: float = None):
        """Return the intracavity field

        Args:
            psi (torch.tensor): The wave function of the gas
            time (float, optional): The time at which to compute the intracavity field. Defaults to None.

        Returns:
            float: The intracavity field :math:`\\alpha`
        """
        order = self. get_order(psi)
        bunching = (torch.abs(psi)**2*self.COS2).sum()
        self._cavity_detuning_tilde = 2*np.pi * \
            self._cavity_detuning(time)-self.c3*bunching
        self.c6 = self.c2-self.c3*bunching

        self.eta = np.sqrt(self._lattice_depth(time))*self.eta_prefactor

        alpha = self.c1*self.eta*order/self.c6

        return alpha

    def get_order(self, psi: torch.tensor):
        """Return the order parameter for self-organization

        Args:
            psi (torch.tensor): The wave function of the gas

        Returns:
            float: The order parameter
        """

        return (torch.abs(psi)**2*self.COS).sum()

    def potential_function(self, X: torch.tensor, Y: torch.tensor, psi: torch.tensor, time: float = None):

        self.c2 = 2*np.pi*self._cavity_detuning(time)+1j*self.kappa
        alpha = self.get_alpha(psi, time)

        self.pump_lattice = self._lattice_depth(time) * self._pump_lattice
        cavity_lattice = torch.abs(alpha)**2 * self._cavity_lattice
        interaction = 2 * torch.sqrt(self._gaussian_profile) / self.gas.adim_pulse * self.eta*self.COS*torch.real(alpha)

        return self.pump_lattice + cavity_lattice + interaction