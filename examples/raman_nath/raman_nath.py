from datetime import datetime
from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Contact, Trap, Lattice
from torchgpe.utils import parse_config

import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib import ticker
from scipy.constants import hbar, codata
from tqdm.auto import tqdm
import scipy.constants as spconsts

def plot_bec(a, lat):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].pcolormesh(a.X.cpu(), a.Y.cpu(), (torch.abs(a.psi)**2).cpu(), vmin=0, shading='auto')
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$")
    plt.colorbar(im0, ax=ax[0], orientation='vertical')

    im = ax[1].pcolormesh(a.Kx.cpu(), a.Ky.cpu(), torch.abs(a.psik).cpu(), shading='auto')
    ax[1].set_ylim([-30, 30])
    ax[1].set_xlim([-30, 30])
    ax[1].set_ylabel(r"$k_y$")
    ax[1].set_xlabel(r"$k_x$")
    k2 = 2*2*np.pi*(a.adim_length/lat.lam) #adimentionalized 2k vector
    circle1=plt.Circle((k2, 0),2, fill=False, edgecolor='red')
    circle2=plt.Circle((-k2, 0),2, fill=False, edgecolor='red')
    ax[1].add_patch(circle1)
    ax[1].add_patch(circle2)

    plt.colorbar(im, ax=ax[1], orientation='vertical')
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")

    plt.suptitle(datetime.now().strftime("%d %b %Y %H:%M:%S"))
    fig.tight_layout()
    plt.savefig("raman_nath.png")

if __name__ == "__main__":
    config = parse_config("configuration.yaml")
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    contact = Contact()
    trap = Trap(**config["potentials"]["trap"])

    bec = Gas(**config["gas"], float_dtype=torch.float32, complex_dtype=torch.complex64)
    bec.psi = torch.exp(-(bec.X**2 + bec.Y**2)/(2*(config["initial_wavefunction"]["gaussian_sigma"] / bec.adim_length)**2))
    bec.ground_state([trap, contact], callbacks=[], **config["propagation"]["imaginary_time"])

    #Apply pulse of TP for 15 us and 10 Erecoil lattice
    lattice = Lattice(V0 = 10, lam=1e-6)
    bec.propagate(final_time=0.000015, time_step=config["propagation"]["real_time"]["time_step"], potentials=[trap, contact, lattice], callbacks=[])
    #Time evolution after the pulse
    bec.propagate(final_time=0.0014, time_step=config["propagation"]["real_time"]["time_step"], potentials=[contact], callbacks=[])

    plot_bec(bec, lattice)
