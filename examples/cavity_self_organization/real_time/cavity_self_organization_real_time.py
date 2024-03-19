from torchgpe.bec2D import Gas
from torchgpe.bec2D.callbacks import CavityMonitor
from torchgpe.bec2D.potentials import Contact, DispersiveCavity, Trap
from torchgpe.utils.potentials import linear_ramp

from torchgpe.utils import parse_config
from datetime import datetime
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

ramp = config["boundaries"]["lattice_ramp"]
detunings = torch.linspace(*config["boundaries"]["cavity_detuning"])
depths = torch.tensor([ramp(t) for t in torch.arange(0, config["propagation"]["real_time"]["final_time"], config["propagation"]["real_time"]["time_step"])])

alphas = torch.tensor(np.empty((detunings.shape[0], depths.shape[0]), dtype=complex))

for d_idx, detuning in enumerate(tqdm(detunings, smoothing=0, desc = "Phase diagram", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
    cavity = DispersiveCavity(lattice_depth=ramp, cavity_detuning=detuning, **config["potentials"]["cavity"])

    cavityMonitor = CavityMonitor(cavity)

    bec.psi = torch.exp(-(bec.X**2 + bec.Y**2)/(2*(config["initial_wavefunction"]["gaussian_sigma"] / bec.adim_length)**2))

    bec.ground_state([trap, contact], callbacks=[], **config["propagation"]["imaginary_time"])

    bec.propagate(potentials = [trap, contact, cavity], callbacks=[cavityMonitor], **config["propagation"]["real_time"])

    alphas[d_idx] = cavityMonitor.alpha[0]

def pi_tick_formatter(val, pos):
    if val == 0: return 0
    if (val/np.pi*4) % 4 == 0:
        return f"${('+' if np.sign(val)==1 else '-') if abs(val/np.pi)==1 else int(val/np.pi)}\\pi$"
    return f"${('+' if np.sign(val)==1 else '-') if abs(val/np.pi*4)==1 else int(val/np.pi*4)}\\pi / 4$"

def plot_pd(x, y, z):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.flatten()
    
    im0 = ax[0].pcolormesh(x, y/1e6, np.log10(np.abs(z)), shading='auto', cmap="viridis")
    ax[0].set_xlabel(r"$V_i$ [$E_r$]")
    ax[0].set_ylabel(r"$\Delta_c$ [$MHz$]")
    plt.colorbar(im0, ax = ax[0], orientation='vertical', label="$\\log|\\alpha|$")

    x_left, x_right = ax[0].get_xlim()
    y_low, y_high = ax[0].get_ylim()
    ax[0].set_aspect(abs((x_right-x_left)/(y_low-y_high)) * 1)
    
    ax[0].axhline(y=config["gas"]["N_particles"]*config["potentials"]["cavity"]["cavity_coupling"]**2/(2*config["potentials"]["cavity"]["atomic_detuning"])/1e6, color="red", ls="dashed")

    im0 = ax[1].pcolormesh(x, y/1e6, np.angle(z), shading='auto', cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax[1].set_xlabel(r"$V_i$ [$E_r$]")
    ax[1].set_ylabel(r"$\Delta_c$ [$MHz$]")
    cbar = plt.colorbar(im0, ax = ax[1], orientation='vertical', label="$Arg(\\alpha)$", format=ticker.FuncFormatter(pi_tick_formatter), ticks=ticker.MultipleLocator(base=np.pi))

    x_left, x_right = ax[1].get_xlim()
    y_low, y_high = ax[1].get_ylim()
    ax[1].set_aspect(abs((x_right-x_left)/(y_low-y_high)) * 1)
    plt.suptitle(datetime.now().strftime("%d %b %Y %H:%M:%S"))

    plt.tight_layout()

    
    plt.savefig("cavity_self_organization_real_time.png")

plot_pd(depths, detunings, alphas)
# ! check meshgrid indexing and plotting