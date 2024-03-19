import torch
import numpy as np

import fcntl
import json
import warnings
import matplotlib.pyplot as plt
import tempfile
from os import path
import ffmpeg
from shutil import rmtree
from abc import ABCMeta
from matplotlib import ticker
from .potentials import DispersiveCavity
from ..utils.potentials import LinearPotential, NonLinearPotential
from ..utils.plotting import pi_tick_formatter
from matplotlib.gridspec import GridSpec
from ..utils import prompt_yes_no, enumerate_chunk
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm.auto import tqdm
from matplotlib.colors import LogNorm
import multiprocess
import atexit
import signal
import psutil
from ..utils.callbacks import Callback


class CavityMonitor(Callback):
    """Callback monitoring the time dependent parameters of a dispersive cavity and its field.

    During the simulation, the values of cavity detuning, pump strength and cavity field are stored. Once the simulation is finished, the saved parameters are accessible via the :py:attr:`gpe.bec2D.callbacks.CavityMonitor.alpha`, :py:attr:`gpe.bec2D.callbacks.CavityMonitor.pump` and :py:attr:`gpe.bec2D.callbacks.CavityMonitor.cavity_detuning` tensors.

    Args:

        dispersive_cavity (DispersiveCavity): The cavity to be monitored.
        save_every (int): Optional. The number of epochs after which the parameters should be saved. Defaults to 1.
    """

    def __init__(self, dispersive_cavity: DispersiveCavity, save_every=1) -> None:
        super().__init__()
        self.save_every = save_every
        #: list(float): A list of the pump strengths. It is a list of lists, where each inner list contains the pump strengths for a single propagation. At the end of the simulation, it is converted to a PyTorch tensor.
        self.pump = []
        #: list(float): A list of the cavity detunings. It is a list of lists, where each inner list contains the cavity detunings for a single propagation. At the end of the simulation, it is converted to a PyTorch tensor.
        self.cavity_detuning = []
        #: list(complex): A list of the cavity field amplitudes. It is a list of lists, where each inner list contains the cavity field amplitudes for a single propagation. At the end of the simulation, it is converted to a PyTorch tensor.
        self.alpha = []
        #: list(float): A list of the times at which the parameters were saved. It is a list of lists, where each inner list contains the times for a single propagation. At the end of the simulation, it is converted to a PyTorch tensor.
        self.times = []
        self.cavity = dispersive_cavity
    
    def on_propagation_begin(self):
        self.alpha.append([])
        self.pump.append([])
        self.cavity_detuning.append([])
        self.times.append([])
    
    def on_epoch_end(self, epoch):
        if epoch % self.save_every != 0:
            return

        time = epoch*self.propagation_params["time_step"]
        self.times[-1].append(time)
        alpha = self.cavity.get_alpha(self.gas.psi, time=time)

        self.alpha[-1].append(alpha)
        self.pump[-1].append(self.cavity._lattice_depth(time))
        self.cavity_detuning[-1].append(self.cavity._cavity_detuning(time))
        
    def on_propagation_end(self):
        self.alpha[-1] = torch.tensor(self.alpha[-1])
        self.pump[-1] = torch.tensor(self.pump[-1])
        self.cavity_detuning[-1] = torch.tensor(self.cavity_detuning[-1])
        self.times[-1] = torch.tensor(self.times[-1])

class Animation(Callback):
    """Callback generating an animation of the propagation of the wavefunction.

    Args:
        output_file (str): The path where to store the mp4 animation.
        save_every (int): Optional. The number of epochs after which a frame of the animation is saved. Defaults to 1.
        fps (int): Optional. The number of frames per second of the animation. Defaults to 25.
        cores (int): Optional. The number of cores to use for the generation of the images. Defaults to 1.
        density (bool): Optional. Whether to plot the real space density. Defaults to True.
        phase (bool): Optional. Whether to plot the phase. Defaults to True.
        densityk (bool): Optional. Whether to plot the momentum space density. Defaults to False.
        potentials (bool): Optional. Whether to plot the potential landscape. Defaults to False.
        cavities (list): Optional. A list of :class:`gpe.bec2D.potentials.DispersiveCavity` objects to monitor. Defaults to [].
        time_dependent_variables (list): Optional. A list of tuples of the form (label, function) where label is a string and function is a function of time returning a float. The value of the function will be plotted as a function of time. Defaults to [].
    """

    def __init__(self, output_file, save_every=1, fps=25, cores=1, density=True, phase=True, densityk=False, potentials=False, cavities=[], time_dependent_variables=[]):

        super().__init__()
        self.save_every = save_every
        self.output_folder = path.dirname(output_file)
        self.output_file = output_file
        self.fps = fps
        self.N_cores = cores
        self._plot_density = density
        self._plot_phase = phase
        self._plot_densityk = densityk
        self._plot_potentials = potentials
        self._time_dependent_variables = time_dependent_variables
        self._tdv_history = [[] for _ in range(len(time_dependent_variables))]
        self._cavities = cavities
        self._cavities_history = [[] for _ in range(len(cavities))]
        self._times = []

        if not path.exists(self.output_folder):
            raise Exception("The output folder does not exist")
        if path.exists(self.output_file):
            if not prompt_yes_no("The specified file already exists. Are you sure you want to overwrite it? Y/n", True):
                raise Exception("The output file already exists")

        N_2d_plots = density + phase + densityk + potentials
        self.N_1d_plots = 2*len(cavities) + len(time_dependent_variables)
        self.n_cols = 2 if N_2d_plots % 2 == 0 else N_2d_plots
        self.n_rows = (int(np.ceil(N_2d_plots/2)) if N_2d_plots %
                       2 == 0 else 1)+self.N_1d_plots
        self._height_ratios = [3]*(self.n_rows-self.N_1d_plots)
        self._height_ratios.extend([1]*self.N_1d_plots)


    def _register_run(self):
        with open(path.join(path.expanduser("~"), ".GPE_animation_cleanup.json"), 'a+') as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # Acquire exclusive lock
            file.seek(0)
            try:
                existing_data = json.load(file)
            except (json.JSONDecodeError, EOFError):
                existing_data = {}
                warnings.warn("The executions register file does not exist and it will be created. If, before now, you have run the animation callback and the process has been interrupted, there might be some leftover temporary folders. Please, check the folder /tmp/ and delete eventual temporary folders manually.")
            
            existing_data.setdefault(str(psutil.Process().pid), []).extend([self.temp_dir])
            file.truncate(0)
            file.seek(0)
            json.dump(existing_data, file)
            file.flush()
            fcntl.flock(file, fcntl.LOCK_UN)  # Release lock

    def _deregister_run(self, pid=None, folder=None):
        if pid is None:
            pid = str(psutil.Process().pid)

        with open(path.join(path.expanduser("~"), ".GPE_animation_cleanup.json"), 'a+') as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # Acquire exclusive lock
            file.seek(0)
            try:
                existing_data = json.load(file)
            except (json.JSONDecodeError, EOFError):
                existing_data = {}
            if pid in existing_data:
                if folder is not None:
                    self.clear_dir(folder)
                    existing_data[pid] = [f for f in existing_data[pid] if f != folder]
                    if len(existing_data[pid]) == 0:
                        del existing_data[pid]
                else:
                    for f in existing_data[pid]:
                        self.clear_dir(f)
                    del existing_data[pid]
            file.truncate(0)
            file.seek(0)
            json.dump(existing_data, file)
            file.flush()
            fcntl.flock(file, fcntl.LOCK_UN)  # Release lock

    def _clean_leftovers(self):
        try: 
            with open(path.join(path.expanduser("~"), ".GPE_animation_cleanup.json"), "r") as f:
                runs = json.load(f)
                for key, value in runs.items():
                    if not psutil.pid_exists(int(key)):
                        self._deregister_run(key)
        except (json.JSONDecodeError, FileNotFoundError):
            return

    def clear_dir(self, dir):  
        if path.exists(dir):
            rmtree(dir)

    def on_propagation_begin(self) -> None:
        """At the beginning of the simulation, creates a temporary folder where to store the images and initializes the variables used to store the data.

        Args:
            epoch (int): The epoch number
        """

        self.temp_dir = tempfile.mkdtemp()

        # Register the temporary folder for deletion at exit and on SIGINT. Check if the folder exists before deleting it to avoid errors
        
        self._clean_leftovers()
        self._register_run()

        atexit.register(self.clear_dir, self.temp_dir)
        signal.signal(signal.SIGINT, lambda sig, frame: (self.clear_dir(self.temp_dir), self._deregister_run(), signal.default_int_handler(signal.SIGINT, None)) )
        signal.signal(signal.SIGTERM, lambda sig, frame: (self.clear_dir(self.temp_dir), self._deregister_run(), signal.default_int_handler(signal.SIGTERM, None)) )

        self.tensor_index = 0
        if self._plot_potentials:
            self._potentials = self.propagation_params["potentials"]
        self._max_density = 0
        self._max_densityk = 0
        self._max_potential = 0

        if self.propagation_params["N_iterations"]/self.save_every > 1000:
            warnings.warn("The animation is going to generate many frames. Consider increasing the save_every parameter.")


    def on_epoch_end(self, epoch):
        """After each epoch, if the epoch number is a multiple of ``save_every``, saves the data to the temporary folder.

        Args:
            epoch (int): The epoch number

        """
        if epoch % self.save_every != 0:
            return

        time = epoch*self.propagation_params["time_step"]
        self._times.append(time*1000)

        if self._plot_density:
            torch.save(self.gas.density.to(torch.float16), path.join(
                self.temp_dir, f"density_{self.tensor_index}.torch"))
            max_density = self.gas.density.max()
            if max_density > self._max_density:
                self._max_density = max_density

        if self._plot_phase:
            torch.save(self.gas.phase.to(torch.float16), path.join(
                self.temp_dir, f"phase_{self.tensor_index}.torch"))

        if self._plot_densityk:
            torch.save(self.gas.densityk.to(torch.float16), path.join(
                self.temp_dir, f"densityk_{self.tensor_index}.torch"))
            max_densityk = self.gas.densityk.max()
            if max_densityk > self._max_densityk:
                self._max_densityk = max_densityk

        if self._plot_potentials:
            total_potential = sum(
                potential.get_potential(self.gas.X, self.gas.Y, time) for potential in self._potentials if issubclass(type(potential), LinearPotential)
            )
            total_potential += sum(
                potential.potential_function(self.gas.X, self.gas.Y, self.gas.psi, time) for potential in self._potentials if issubclass(type(potential), NonLinearPotential)
            )
            torch.save(total_potential.to(torch.float16), path.join(
                self.temp_dir, f"potential_{self.tensor_index}.torch"))
            max_potential = total_potential.max()
            if max_potential > self._max_potential:
                self._max_potential = max_potential

        if len(self._cavities):
            for i, cavity in enumerate(self._cavities):
                self._cavities_history[i].append(
                    cavity.get_alpha(self.gas.psi, time=time).cpu())

        if len(self._time_dependent_variables):
            for i, [label, variable] in enumerate(self._time_dependent_variables):
                self._tdv_history[i].append(variable(time))

        self.tensor_index += 1

    def _plot_stored(self, params) -> None:
        """Plots the data stored in the temporary folder. 

        Args:
            params (list): A list of tuples of the form (image_index, time) where image_index is the index of the image to be plotted and time is the in-simulation time at which the image was saved.
        """
        for param in params:
            image_index, time = param
            fig = plt.figure(
                figsize=(6*self.n_cols, 6*self.n_rows-4*self.N_1d_plots))
            gs = GridSpec(self.n_rows, self.n_cols, figure=fig,
                          height_ratios=self._height_ratios)

            row_index = 0
            col_index = 0

            if self._plot_density:
                ax = fig.add_subplot(gs[row_index, col_index])
                im = ax.pcolormesh(self.gas.X.cpu(), self.gas.Y.cpu(), torch.load(path.join(
                    self.temp_dir, f"density_{image_index}.torch"), map_location="cpu"), norm=LogNorm(vmax=self._max_density, vmin=1e-10), shading='auto')
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="6%", pad="2%")
                cbar = plt.colorbar(im, cax=cax)
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")
                ax.set_title("Real space density")
                ax.set_aspect('equal')
                if col_index == self.n_cols-1:
                    col_index = 0
                    row_index += 1
                else:
                    col_index += 1
            if self._plot_phase:
                ax = fig.add_subplot(gs[row_index, col_index])
                im = ax.pcolormesh(self.gas.X.cpu(), self.gas.Y.cpu(), torch.load(path.join(
                    self.temp_dir, f"phase_{image_index}.torch"), map_location="cpu"), vmin=-np.pi, vmax=np.pi, shading='auto', cmap="bwr")
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="6%", pad="2%")
                cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(
                    pi_tick_formatter), ticks=ticker.MultipleLocator(base=np.pi/2))
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")  # account for units and set title
                ax.set_title("Phase")
                ax.set_aspect('equal')
                if col_index == self.n_cols-1:
                    col_index = 0
                    row_index += 1
                else:
                    col_index += 1
            if self._plot_densityk:
                ax = fig.add_subplot(gs[row_index, col_index])
                im = ax.pcolormesh(self.gas.Kx.cpu(), self.gas.Ky.cpu(), torch.load(path.join(
                    self.temp_dir, f"densityk_{image_index}.torch"), map_location="cpu"), norm=LogNorm(vmax=self._max_densityk, vmin=1e-10), shading='auto')
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="6%", pad="2%")
                cbar = plt.colorbar(im, cax=cax)
                ax.set_xlabel(r"$kx$")
                ax.set_ylabel(r"$ky$")  # account for units and set title
                ax.set_title("Momentum space density")
                ax.set_aspect('equal')
                if col_index == self.n_cols-1:
                    col_index = 0
                    row_index += 1
                else:
                    col_index += 1
            if self._plot_potentials:
                ax = fig.add_subplot(gs[row_index, col_index])
                im = ax.pcolormesh(self.gas.X.cpu(), self.gas.Y.cpu(), torch.load(path.join(
                    self.temp_dir, f"potential_{image_index}.torch"), map_location="cpu"), shading='auto', vmin=0, vmax=self._max_potential)
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="6%", pad="2%")
                cbar = plt.colorbar(im, cax=cax)
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")  # account for units and set title
                ax.set_title("Potential landscape")
                ax.set_aspect('equal')
                if col_index == self.n_cols-1:
                    col_index = 0
                    row_index += 1
                else:
                    col_index += 1

            if len(self._cavities):
                for i, history in enumerate(self._cavities_history, start=1):
                    ax = fig.add_subplot(gs[row_index, :])
                    ax.axvline(x=self._times[image_index], color="red")
                    ax.plot(self._times, np.abs(history))
                    ax.set_xlabel(r"$t$ [$ms$]")
                    ax.set_ylabel(r"$|\alpha|$")  # account for units and title
                    ax.set_title(f"Cavity {i} field")
                    ax.set_xlim(0, self._times[-1])
                    col_index = 0
                    row_index += 1

                    ax = fig.add_subplot(gs[row_index, :])
                    ax.axvline(x=self._times[image_index], color="red")
                    ax.plot(self._times, np.angle(history))
                    ax.set_xlabel(r"$t$ [$ms$]")
                    # account for units and title
                    ax.set_ylabel(r"$Arg(\alpha)$")
                    ax.set_xlim(0, self._times[-1])
                    col_index = 0
                    row_index += 1

            if len(self._time_dependent_variables):
                for i, [label, variable] in enumerate(self._time_dependent_variables):
                    ax = fig.add_subplot(gs[row_index, :])
                    ax.axvline(x=self._times[image_index], color="red")
                    ax.plot(self._times, self._tdv_history[i])
                    ax.set_xlabel(r"$t$ [$ms$]")
                    ax.set_ylabel(r"$V$")  # account for units and title
                    ax.set_title(label)
                    ax.set_xlim(0, self._times[-1])
                    col_index = 0
                    row_index += 1

            if isinstance(self.propagation_params["time_step"], float):
                fig.suptitle(
                    f"t: {time:.2f} ms")

            plt.subplots_adjust(left=0.1, bottom=0.1,
                                right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            fig.savefig(path.join(self.temp_dir, f"{image_index}.png"))
            plt.close(fig)

    def on_propagation_end(self) -> None:
        """At the end of the simulation, generates the animation and saves it to the specified path.
        """
        ctx = multiprocess.get_context("spawn")

        with ctx.Pool(self.N_cores) as pool:
            _ = list(tqdm(pool.imap(self._plot_stored, enumerate_chunk(self._times, int(np.ceil(len(self._times)/self.N_cores)))),
                     total=self.N_cores, smoothing=0, desc="Picture generation", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', leave=False))

        print("Merging the pictures into a movie...", end="\r")
        (
            ffmpeg
            .input(path.join(self.temp_dir, "%d.png"), framerate=self.fps)
            .output(self.output_file, pix_fmt='yuv420p', vcodec='libx264')
            .global_args("-nostats")
            .global_args("-loglevel", "0")
            .run(overwrite_output=True)
        )

        print(f"Animation saved to {self.output_file}")
        self.clear_dir(self.temp_dir)
        self._deregister_run(folder=self.temp_dir)
