from abc import ABCMeta, abstractmethod
import torch


# --- Time dependent parameters ---


def time_dependent_variable(var):
    """Transform a variable into a function of time

    Args:
        var (Union[float, Callable]): The variable to transform. If it is a function, it is returned as is. If it is a constant, it is transformed into a function that returns the constant.
    
    Examples:
        >>> time_dependent_variable(1)
        lambda _: 1
        >>> time_dependent_variable(lambda t: t)
        lambda t: t
    """
    return var if callable(var) else (lambda _: var)


def any_time_dependent_variable(*vars):
    """Check if any of the variables is time dependent

    Args:
        *vars (Union[float, Callable]): The variables to check. If any of them is a function, the function returns True. If all of them are constants, the function returns False.
    
    Examples:
        >>> any_time_dependent_variable(1, 2, 3)
        False
        >>> any_time_dependent_variable(1, lambda t: t, 3)
        True
    """
    return any(map(callable, vars))

# --- Common behaviours in time ---


def linear_ramp(v0=0, t0=0, v1=1, t1=1):
    """Implements a linear ramp from :math:`v_0` to :math:`v_1` between :math:`t_0` and :math:`t_1`. The ramp is constant outside of the interval.

    Args:
        v0 (float, optional): The initial value of the ramp. Defaults to :math:`0`.
        t0 (float, optional): The initial time of the ramp. Defaults to :math:`0`.
        v1 (float, optional): The final value of the ramp. Defaults to :math:`1`.
        t1 (float, optional): The final time of the ramp. Defaults to :math:`1`.

    Returns:
        callable: A function that returns the value of the ramp at time :math:`t`
        """
    return lambda t: v0 if t < t0 else v0 + (v1-v0)*(t-t0)/(t1-t0) if t < t1 else v1


def s_ramp(v0=0, t0=0, v1=1, t1=1):
    """Implements a smooth ramp from :math:`v_0` to :math:`v_1` between :math:`t_0` and :math:`t_1`. The ramp is constant outside of the interval.

    Args:
        v0 (float, optional): The initial value of the ramp. Defaults to :math:`0`.
        t0 (float, optional): The initial time of the ramp. Defaults to :math:`0`.
        v1 (float, optional): The final value of the ramp. Defaults to :math:`1`.
        t1 (float, optional): The final time of the ramp. Defaults to :math:`1`.

    Returns:
        callable: A function that returns the value of the ramp at time :math:`t`
        """
    return lambda t: v0 if t < t0 else v0 - 2*(v1-v0)*((t-t0)/(t1-t0))**3 + 3*(v1-v0)*((t-t0)/(t1-t0))**2 if t < t1 else v1


def quench(v0=1, v1=0, quench_time=1):
    """Implements a quench from :math:`v_0` to :math:`v_1` at :math:`t=quench_time`. The value is constant outside of the interval.

    Args:
        v0 (float, optional): The initial value. Defaults to :math:`1`.
        v1 (float, optional): The final value. Defaults to :math:`0`.
        quench_time (float, optional): The time at which the quench occurs. Defaults to :math:`1`.

    Returns:
        float: The value of the quench at time :math:`t`
        """
    return lambda t: v0 if t < quench_time else v1

# --- Potential base classes ---


class Potential(metaclass=ABCMeta):
    """Base class for potentials. It is not meant to be used directly, but to be inherited by other classes.
    """

    def __init__(self):
        #: bool: Whether the potential is time dependent or not
        self.is_time_dependent = False
        #: gpe.bec2D.gas.Gas: The :class:`~gpe.bec2D.gas.Gas` object to which the potential is applied
        self.gas = None

    def set_gas(self, gas):
        """Set the :class:`~gpe.bec2D.gas.Gas` object to which the potential is applied

        Args:
            gas (Gas): The :class:`~gpe.bec2D.gas.Gas` object to which the potential is applied
        """
        self.gas = gas

    def on_propagation_begin(self):
        """Called at the beginning of the propagation. It is used to post-process the parameters of the potential, once the :class:`~gpe.bec2D.gas.Gas` object has been set."""
        pass


class LinearPotential(Potential, metaclass=ABCMeta):
    """Base class for linear 2D potentials. It is not meant to be used directly, but to be inherited by other classes.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_potential(self, X: torch.tensor, Y: torch.tensor, time: float = None):
        """Return the linear potential evaluated on the grid. If time dependent parameters are present, the parameter ``time`` is also specified, otherwise it is set to ``None``. 

        Args:
            X (torch.tensor): The X coordinates on the adimentionalized grid where to compute the potential.
            Y (torch.tensor): The Y coordinates on the adimentionalized grid where to compute the potential.
            time (float, optional): If time dependent parameters are specified, the time at which to evaluate the potential. Defaults to None.

        Returns:
            torch.tensor: The potential evaluated on the grid.
        """
        pass


class NonLinearPotential(Potential, metaclass=ABCMeta):
    """Base class for non-linear 2D potentials. It is not meant to be used directly, but to be inherited by other classes.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def potential_function(self, X: torch.tensor, Y: torch.tensor, psi: torch.tensor, time: float = None):
        """Return the non-linear potential evaluated on the grid. If time dependent parameters are present, the parameter ``time`` is also specified, otherwise it is set to ``None``. 

        Args:
            X (torch.tensor): The X coordinates on the adimentionalized grid where to compute the potential.
            Y (torch.tensor): The Y coordinates on the adimentionalized grid where to compute the potential.
            psi (torch.tensor): The wave function of the gas.
            time (float, optional): If time dependent parameters are specified, the time at which to evaluate the potential. Defaults to None.

        Returns:
            torch.tensor: The potential evaluated on the grid.
        """

        pass
