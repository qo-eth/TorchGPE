from typing import List, Union
import torch
import numpy as np
from torch.nn.functional import pad
import scipy.constants as spconsts

from .potentials import Zero
from ..utils import normalize_wavefunction, ftn, iftn
from ..utils.elements import elements_dict
from ..utils.potentials import Potential
from ..utils.propagation import imaginary_time_propagation, real_time_propagation
from ..utils.callbacks import Callback

UPDATED_PSI = 0
UPDATED_PSIK = 1
UPDATED_BOTH = 2


class Gas():
    """Quantum gas.

    The parameters :py:attr:`N_grid` and :py:attr:`grid_size` specify a computational grid on which the wavefunction
    is defined and evolved. :class:`Gas` exposes methods to perform real time propagation and to compute the ground 
    state's wave function via imaginary time propagation.

    Args:
        element (str): Optional. The element the gas is made of. Defaults to "87Rb".
        N_particles (int): Optional. The number of particles in the gas. Defaults to :math:`10^6`.
        N_grid (int): Optional. The number of points on each side of the computational grid. Defaults to :math:`2^8`.
        grid_size (float): Optional. The side of the computational grid. Defaults to :math:`10^{-6}`.
        device (torch.device or None): Optional. The device where to store tensors. Defaults to None, meaning that GPU will be used if available.
        float_dtype (:py:attr:`torch.dtype`): Optional. The dtype used to represent floating point numbers. Defaults to :py:attr:`torch.double`.
        complex_dtype (:py:attr:`torch.dtype`): Optional. The dtype used to represent complex numbers. Defaults to :py:attr:`torch.complex128`.
        adimensionalization_length (float): Optional. The unit of length to be used during the simulations. Defaults to :math:`10^{-6}`.
    """

    def __init__(self, element: str = "87Rb", N_particles: int = int(1e6),
                 N_grid: int = 2**8, grid_size: float = 1e-6,
                 device: Union[torch.device, None] = None, float_dtype: torch.dtype = torch.double, complex_dtype: torch.dtype = torch.complex128, adimensionalization_length: float = 1e-6) -> None:

        #: str: The element the gas is made of.
        self.element = element
        #: float: The mass of the gas. This is automatically derived from the :py:obj:`~Gas.element` parameter.
        self.mass = elements_dict[self.element]["m"] 
        #: float: The pulse of the :math:`d_2` line. This is automatically derived from the :py:obj:`~Gas.element` parameter.
        self.d2_pulse = elements_dict[self.element]["omega d2"]

        if (N_particles != int(N_particles)):
            raise TypeError("The number of particles must be an integer")
        #: int: The number of particles in the gas.
        self.N_particles = int(N_particles)

        # If no custom device has been specified, use GPU if available
        #: torch.device: The device where to store tensors.
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        #: torch.dtype: The dtype used to represent floating point numbers.
        self.float_dtype = float_dtype
        #: torch.dtype: The dtype used to represent complex numbers.
        self.complex_dtype = complex_dtype

        # Adimensionalization length and pulse to do calculations with pure numbers
        #: float: Adimensionalization length used to work with pure numbers.
        self.adim_length = adimensionalization_length
        #: float: Adimensionalization pulse used to work with pure numbers. Its value is :math:`\frac{\hbar}{m l^2}`, where :math:`m` is the mass and :math:`l` the adimensionalization length.
        self.adim_pulse = spconsts.hbar/(self.mass*self.adim_length**2)

        # Create the grid in adimenisonalized units
        #: float: The side of the computational grid along the :mathx axis in adimensionalized units.
        self.grid_size_x = grid_size / self.adim_length
        #: float: The side of the computational grid along the y axis in adimensionalized units.
        self.grid_size_y = grid_size / self.adim_length
        #: int: The number of points on each side of the computational grid.
        self.N_grid = int(N_grid)

        # Grid in real space
        # The grid is centered in 0, with a total side length of grid_size/adim_length.
        # The total grid is made of N_grid**2 points
        
        #: torch.Tensor: The vector of adimensionalized grid coordinates along the :math:`x` axis.
        self.x = torch.linspace(-self.grid_size_x/2, self.grid_size_x/2,
                                self.N_grid, dtype=self.float_dtype, device=self.device)
        #: torch.Tensor: The vector of adimensionalized grid coordinates along the :math:`y` axis.
        self.y = torch.linspace(-self.grid_size_y/2, self.grid_size_y/2,
                                self.N_grid, dtype=self.float_dtype, device=self.device)
        #: float: The distance between two consecutive points of the grid along the :math:`x` axis in adimensionalized units.
        self.dx = self.x[1]-self.x[0]
        #: float: The distance between two consecutive points of the grid along the :math:`y` axis in adimensionalized units.
        self.dy = self.y[1]-self.y[0]
        coordinates = torch.meshgrid(self.x, self.y, indexing="xy")
        #: torch.Tensor: The matrix of :math:`x` coordinates of the grid in adimensionalized units.
        self.X = coordinates[0]
        #: torch.Tensor: The matrix of :math:`y` coordinates of the grid in adimensionalized units.
        self.Y = coordinates[1]
        del coordinates

        # Grid in momentum space
        #: torch.Tensor: The vector of adimensionalized momenta along the :math:`kx` axis.
        self.kx = 2*np.pi * torch.fft.fftshift(torch.fft.fftfreq(
            self.N_grid + 2 * (self.N_grid//2), self.dx, dtype=self.float_dtype, device=self.device))
        #: torch.Tensor: The vector of adimensionalized momenta along the :math:`ky` axis.
        self.ky = 2*np.pi * torch.fft.fftshift(torch.fft.fftfreq(
            self.N_grid + 2 * (self.N_grid//2), self.dy, dtype=self.float_dtype, device=self.device))
        #: float: The distance between two consecutive points of the grid along the :math:`kx` axis in adimensionalized units.
        self.dkx = self.kx[1] - self.kx[0]
        #: float: The distance between two consecutive points of the grid along the :math:`ky` axis in adimensionalized units.
        self.dky = self.ky[1] - self.ky[0]
        momenta = torch.meshgrid(self.kx, self.ky, indexing="xy")
        #: torch.Tensor: The matrix of :math:`kx` coordinates of the grid in adimensionalized units.
        self.Kx = momenta[0]
        #: torch.Tensor: The matrix of :math:`ky` coordinates of the grid in adimensionalized units.
        self.Ky = momenta[1]
        del momenta

        # Create the wave functions
        self._psi = torch.zeros_like(self.X)
        self._psik = torch.zeros_like(self.Kx)
        # Specifies the last updated wave function (psi or psik)
        self._updated_wavefunction = None

    def ground_state(self, potentials: List[Potential] = [], time_step: complex = -1e-6j, N_iterations: int = int(1e3), callbacks: List[Callback] = [], leave_progress_bar=True):
        """Compute the ground state's wave function.

        Use the split-step Fourier method with imaginary time propagation (ITP) to compute the ground state's wave function of the gas. 
        The potentials acting on the system are specified via the :py:attr:`potentials` parameter. 

        Args:
            potentials (List[:class:`~gpe.utils.potentials.Potential`]): Optional. The list of potentials acting on the system. Defaults to [].
            time_step (complex): Optional. The time step to be used in the ITP. Defaults to :math:`-10^{-6}\,i`.
            N_iterations (int): Optional. The number of steps of ITP to perform. Defaults to :math:`10^{3}`.
            callbacks (List[:class:`~gpe.utils.callbacks.Callback`]): Optional. List of callbacks to be evaluated during the evolution. Defaults to [].
            leave_progress_bar (bool): Optional. Whether to leave the progress bar on screen after the propagation ends. Defaults to True.

        Raises:
            Exception: If time dependent potentials are specified 
            Exception: If the time step is not a purely imaginary number
            Exception: If the imaginary part of the time step is not positive
            Exception: If neither the wave function in real space nor in the one in momentum space have been initialized
        """
        
        # Initial setup of the potentials
        for potential in potentials:
            potential.set_gas(self)
            potential.on_propagation_begin()

        # --- Process parameters ---

        if any(potential.is_time_dependent for potential in potentials):
            raise Exception(
                "Time dependent potentials can't be used in imaginary time propagation")

        if time_step.real != 0:
            raise Exception(
                "Imaginary time propagation requires a purely imaginary time step")
        if np.imag(time_step) >= 0:
            raise Exception(
                "The imaginary part of the time step must be negative")

        if self._updated_wavefunction is None:
            raise Exception(
                "The initial wave function must be initialized by either setting the psi or psik attributes")

        N_iterations = int(N_iterations)

        # Adimensionalize the time_step
        adim_time_step = time_step * self.adim_pulse

        # If no potential has been specified, use an identically zero one
        if len(potentials) == 0:
            potentials = [Zero(None)]

        # Generate a dictionary of runtime settings for the simulations to be given
        # to the callbacks. This list is not complete at the moment
        propagation_parameters = {
            "potentials": potentials,
            "time_step": time_step,
            "N_iterations": N_iterations,
        }

        # Initial setup of the callbacks
        for callback in callbacks:
            callback.set_gas(self)
            callback.set_propagation_params(propagation_parameters)

        imaginary_time_propagation(
            self, potentials, adim_time_step, N_iterations, callbacks, leave_progress_bar)

    def propagate(self, final_time: float, time_step: float = 1e-6, potentials: List[Potential] = [], callbacks: List[Callback] = [], leave_progress_bar=True):
        """Propagate the wave function in real time.

        Use the split-step Fourier method with real time propagation (RTP) to propagate the gas wave function to :py:attr:`final_time`. 
        The potentials acting on the system are specified via the :py:attr:`potentials` parameter. 

        Note:
            The time step is adjusted such that :py:attr:`final_time` is always reached.

        Args:
            final_time (float): The final time up to which the wave function whould be propagated.
            time_step (float): Optional. The time step to be used in the RTP. Defaults to :math:`10^{-6}`.
            potentials (List[:class:`~gpe.utils.potentials.Potential`]): Optional. The list of potentials acting on the system. Defaults to [].
            callbacks (List[:class:`~gpe.utils.callbacks.Callback`]): Optional. List of callbacks to be evaluated during the evolution. Defaults to [].
            leave_progress_bar (bool): Optional. Whether to leave the progress bar on screen after the propagation ends. Defaults to True.

        Raises:
            Exception: If the time step is not a floating point number
            Exception: If the time step is not positive
            Exception: If neither the wave function in real space nor in the one in momentum space have been initialized
        """

        # Initial setup of the potentials
        for potential in potentials:
            potential.set_gas(self)
            potential.on_propagation_begin()

        # --- Process parameters ---

        if not issubclass(type(time_step), (float, )):
            raise Exception(
                "The provided time step is not a floating point number.")
        if time_step <= 0:
            raise Exception("Propagation requires a positive time step")

        # Adjust the time step such that the final time is always reached
        N_iterations = round(final_time/time_step)
        time_step = final_time/N_iterations

        # Array of times to be passed to the time dependent potentials
        times = torch.linspace(0, final_time, N_iterations)

        # Adimensionalize the time_step
        adim_time_step = time_step * self.adim_pulse

        if self._updated_wavefunction is None:
            raise Exception(
                "The initial wave function must be initialized by setting either the psi or psik attributes")

        # If no potential has been specified, use an identically zero one
        if len(potentials) == 0:
            potentials = [Zero(None)]

        # Generate a dictionary of runtime settings for the simulations to be given
        # to the callbacks. This list is not complete at the moment
        propagation_parameters = {
            "potentials": potentials,
            "time_step": time_step,
            "N_iterations": N_iterations,
            "final_time": final_time
        }

        # Initial setup of the callbacks
        for callback in callbacks:
            callback.set_gas(self)
            callback.set_propagation_params(propagation_parameters)

        real_time_propagation(
            self, potentials, adim_time_step, times, callbacks, leave_progress_bar)

    @property
    def density(self):
        """The density of the gas in real space
        """
        return torch.abs(self.psi)**2

    @property
    def densityk(self):
        """The density of the gas in momentum space
        """
        return torch.abs(self.psik)**2

    @property
    def phase(self):
        """The phase (in radians) of the real space wave function
        """
        return torch.angle(self.psi)

    # --- Manage the update of psi and psik ---

    @property
    def psi(self):
        """The real space wave function of the gas.

        Returns the most updated real space wave function of the gas. If the last updated wave function is the one in momentum space, 
        computes and stores the real space wave function as its iFFT before returning it. 
        When a value is assigned to psi, takes care of the normalization before storing it.
        """

        # If the last updated wave function is psik, compute psi
        if self._updated_wavefunction == UPDATED_PSIK:
            # Take into account that psik is padded
            self.psi = iftn(self._psik)[
                self.N_grid//2:self.N_grid+self.N_grid//2, self.N_grid//2:self.N_grid+self.N_grid//2]
            self._updated_wavefunction = UPDATED_BOTH
        return self._psi

    @psi.setter
    def psi(self, value):
        if value.dtype != self.complex_dtype:
            value = value.type(self.complex_dtype)
        self._psi = normalize_wavefunction(value, self.dx, self.dy)
        self._updated_wavefunction = UPDATED_PSI

    @property
    def psik(self):
        """The momentum space wave function of the gas.

        Returns the most updated momentum space wave function of the gas. If the last updated wave function is the one in real space, 
        computes and stores the momentum space wave function as its iFFT before returning it. 
        When a value is assigned to psik, takes care of the normalization before storing it.
        """

        # If the last updated wave function is psi, compute psik
        if self._updated_wavefunction == UPDATED_PSI:
            # Before computing the FFT, pad psik
            self.psik = ftn(pad(self._psi, (self.N_grid//2, self.N_grid//2,
                            self.N_grid//2, self.N_grid//2), mode="constant", value=0))
            self._updated_wavefunction = UPDATED_BOTH
        return self._psik

    @psik.setter
    def psik(self, value):
        if value.dtype != self.complex_dtype:
            value = value.type(self.complex_dtype)
        self._psik = normalize_wavefunction(value, self.dkx, self.dky)
        self._updated_wavefunction = UPDATED_PSIK


    @property
    def coordinates(self):
        """The coordinates of the gas

        Returns a tuple containing the coordinates of the gas in real space.
        """
        return (self.X, self.Y)
    
    @property
    def momenta(self):
        """The momenta of the gas

        Returns a tuple containing the momenta of the gas in momentum space.
        """
        return (self.Kx, self.Ky)
