from .potentials import LinearPotential, NonLinearPotential
import torch
from tqdm.auto import tqdm, trange


def imaginary_time_propagation(gas, potentials, time_step, N_iterations, callbacks, leave_progress_bar=True):
    """Performs imaginary time propagation of a wave function.

    Args:
        gas (Gas): The gas whose wave function has to be propagated.
        potentials (list): The list of potentials to apply.
        time_step (float): The time step to use.
        N_iterations (int): The number of iterations to perform.
        callbacks (list): The list of callbacks to call at the end of each iteration.
        leave_progress_bar (bool, optional): Whether to leave the progress bar after the propagation is complete. Defaults to True.
    """
    # Divide the potentials in linear and nonlinear to precompute the linear ones
    linear_potentials = [potential for potential in potentials if issubclass(
        type(potential), LinearPotential)]
    nonlinear_potentials = [potential for potential in potentials if issubclass(
        type(potential), NonLinearPotential)]

    for callback in callbacks:
        callback.on_propagation_begin()

    # Precompute kinetic propagator and the total linear potential
    kinetic = 0.5 * sum(momentum**2 for momentum in gas.momenta)
    kinetic_propagator = torch.exp(-0.5j * kinetic * time_step)
    total_linear_potential = sum(potential.get_potential(*gas.coordinates) for potential in linear_potentials)

    # Create a progress bar to monitor the evolution
    pbar = trange(N_iterations, smoothing=0, desc="Ground state",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=leave_progress_bar)

    for epoch in pbar:
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # One step of the split-step Fourier method
        propagation_step(gas, total_linear_potential, [],
                         nonlinear_potentials, [], kinetic_propagator, time_step)

        for callback in callbacks:
            callback.on_epoch_end(epoch)

    for callback in callbacks:
        callback.on_propagation_end()


def real_time_propagation(gas, potentials, time_step, times, callbacks, leave_progress_bar=True):
    """Performs real time propagation of a wave function.

    Args:
        gas (Gas): The gas whose wave function has to be propagated.
        potentials (list): The list of potentials to apply.
        time_step (float): The time step to use.
        times (list): The list of times to propagate to.
        callbacks (list): The list of callbacks to call at the end of each iteration.
        leave_progress_bar (bool, optional): Whether to leave the progress bar after the propagation is complete. Defaults to True.
    """
    # Divide the potentials in linear and nonlinear, time dependent and time independent to precompute the static linear ones
    static_linear_potentials = [potential for potential in potentials if issubclass(
        type(potential), LinearPotential) and not potential.is_time_dependent]
    dynamic_linear_potentials = [potential for potential in potentials if issubclass(
        type(potential), LinearPotential) and potential.is_time_dependent]
    static_nonlinear_potentials = [potential for potential in potentials if issubclass(
        type(potential), NonLinearPotential) and not potential.is_time_dependent]
    dynamic_nonlinear_potentials = [potential for potential in potentials if issubclass(
        type(potential), NonLinearPotential) and potential.is_time_dependent]

    for callback in callbacks:
        callback.on_propagation_begin()

    # Precompute kinetic propagator and the total static linear potential
    kinetic = 0.5 * sum(momentum**2 for momentum in gas.momenta)
    kinetic_propagator = torch.exp(-0.5j * kinetic * time_step)
    total_static_linear_potential = sum(potential.get_potential(*gas.coordinates) for potential in static_linear_potentials)

    # Create a progress bar to monitor the evolution
    pbar = tqdm(times, smoothing=0, desc="Propagation",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=leave_progress_bar)

    for epoch, t in enumerate(pbar):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # One step of the split-step Fourier method
        propagation_step(gas, total_static_linear_potential, dynamic_linear_potentials,
                         static_nonlinear_potentials, dynamic_nonlinear_potentials, kinetic_propagator, time_step, t)

        for callback in callbacks:
            callback.on_epoch_end(epoch)

    for callback in callbacks:
        callback.on_propagation_end()


def propagation_step(gas, total_static_linear_potential, dynamic_linear_potentials, static_nonlinear_potentials, dynamic_nonlinear_potentials, kinetic_propagator, time_step, time=None):
    """Performs one step of the split-step Fourier method.

    Args:
        gas (Gas): The gas whose wave function has to be propagated.
        total_static_linear_potential (torch.Tensor): The total static linear potential.
        dynamic_linear_potentials (list): The list of dynamic linear potentials.
        static_nonlinear_potentials (list): The list of static nonlinear potentials.
        dynamic_nonlinear_potentials (list): The list of dynamic nonlinear potentials.
        kinetic_propagator (torch.Tensor): The kinetic propagator.
        time_step (float): The time step to use.
        time (float, optional): The in-simulation time . Defaults to None.
    """
    gas.psik *= kinetic_propagator
    gas.psi *= potential_propagator(gas, time_step, total_static_linear_potential,
                                    dynamic_linear_potentials, static_nonlinear_potentials, dynamic_nonlinear_potentials, time)
    gas.psik *= kinetic_propagator


def potential_propagator(gas, time_step, total_static_linear_potential, dynamic_linear_potentials, static_nonlinear_potentials, dynamic_nonlinear_potentials, time):
    """Computes the potential propagator.

    Args:
        gas (Gas): The gas whose wave function has to be propagated.
        time_step (float): The time step to use.
        total_static_linear_potential (torch.Tensor): The total static linear potential.
        dynamic_linear_potentials (list): The list of dynamic linear potentials.
        static_nonlinear_potentials (list): The list of static nonlinear potentials.
        dynamic_nonlinear_potentials (list): The list of dynamic nonlinear potentials.
        time (float): The in-simulation time.
    """
    # Compute the static nonlinear potential and both the dynamic ones
    total_static_nonlinear_potential = sum(potential.potential_function(
        *gas.coordinates, gas.psi) for potential in static_nonlinear_potentials)
    total_dynamic_linear_potential = sum(potential.get_potential(
        *gas.coordinates, time) for potential in dynamic_linear_potentials)
    total_dynamic_nonlinear_potential = sum(potential.potential_function(
        *gas.coordinates, gas.psi, time) for potential in dynamic_nonlinear_potentials)

    # Compute the propagator due to all the potentials
    return torch.exp(-1j * (total_static_linear_potential + total_static_nonlinear_potential + total_dynamic_linear_potential + total_dynamic_nonlinear_potential) * time_step)
