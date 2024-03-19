from datetime import datetime 
from torchgpe.bec2D import Gas
from torchgpe.bec2D.callbacks import CavityMonitor
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.utils.potentials import linear_ramp
from torchgpe.utils import parse_config

import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib import ticker
from scipy.constants import hbar
from tqdm.auto import tqdm

config = parse_config("configuration.yaml")

np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

contact = Contact()
trap = Trap(**config["potentials"]["trap"])

bec = Gas(**config["gas"], float_dtype=torch.float32, complex_dtype=torch.complex64)

detunings = torch.linspace(*config["boundaries"]["cavity_detuning"])
depths = torch.linspace(*config["boundaries"]["lattice_depth"])

alphas = torch.tensor(np.empty((detunings.shape[0], depths.shape[0]), dtype=complex))

for d_idx, detuning in enumerate(tqdm(detunings, smoothing=0, desc = "Phase diagram", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
    for p_idx, pump in enumerate(tqdm(depths, smoothing=0, desc = "Row", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False)):
        cavity = DispersiveCavity(lattice_depth=pump, cavity_detuning=detuning, **config["potentials"]["cavity"])

        bec.psi = torch.exp(-(bec.X**2 + bec.Y**2)/(2*(config["initial_wavefunction"]["gaussian_sigma"] / bec.adim_length)**2))

        bec.ground_state([trap, contact, cavity], callbacks=[], **config["propagation"]["imaginary_time"])

        alphas[d_idx, p_idx] = cavity.get_alpha(bec.psi)

def pi_tick_formatter(val, pos):
    if val == 0: return 0
    if (val/np.pi*4) % 4 == 0:
        return f"${('+' if np.sign(val)==1 else '-') if abs(val/np.pi)==1 else int(val/np.pi)}\\pi$"
    return f"${('+' if np.sign(val)==1 else '-') if abs(val/np.pi*4)==1 else int(val/np.pi*4)}\\pi / 4$"

def plot_pd(x,y,z):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.flatten()
    
    im0 = ax[0].pcolormesh(x, y/1e6, np.log10(np.abs(z)), shading='auto', cmap="viridis")
    ax[0].set_xlabel(r"$V_i$ [$E_r$]")
    ax[0].set_ylabel(r"$\Delta_c$ [$MHz$]")
    plt.colorbar(im0, ax = ax[0], orientation='vertical', label="$\\log|\\alpha|$")

    x_left, x_right = ax[0].get_xlim()
    y_low, y_high = ax[0].get_ylim()
    ax[0].set_aspect(abs((x_right-x_left)/(y_low-y_high)) * 1)
  
    im0 = ax[1].pcolormesh(x, y/1e6, np.angle(z)%np.pi, shading='auto', cmap="twilight", vmin=0, vmax=np.pi)
    ax[1].set_xlabel(r"$V_i$ [$E_r$]")
    ax[1].set_ylabel(r"$\Delta_c$ [$MHz$]")
    cbar = plt.colorbar(im0, ax = ax[1], orientation='vertical', label="$Arg(\\alpha)$", format=ticker.FuncFormatter(pi_tick_formatter), ticks=ticker.MultipleLocator(base=np.pi))

    x_left, x_right = ax[1].get_xlim()
    y_low, y_high = ax[1].get_ylim()
    ax[1].set_aspect(abs((x_right-x_left)/(y_low-y_high)) * 1)
    plt.suptitle(datetime.now().strftime("%d %b %Y %H:%M:%S"))

    plt.tight_layout()
    
    plt.savefig("blue_self_organization_imaginary_time.png")

plot_pd(depths, detunings, alphas)
