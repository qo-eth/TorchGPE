from torchgpe.bec2D import Gas
from torchgpe.bec2D.potentials import Contact, Trap
from torchgpe.utils import parse_config

import numpy as np
import torch
import matplotlib.pyplot as plt 
from scipy.constants import hbar, physical_constants
from datetime import datetime 

a_bohr = physical_constants["Bohr radius"][0]

config = parse_config("configuration.yaml")
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

contact = Contact(**config["potentials"]["contact"])
trap = Trap(**config["potentials"]["trap"])

bec = Gas(**config["gas"], float_dtype=torch.float32, complex_dtype=torch.complex64)
bec.psi = torch.exp(-(bec.X**2 + bec.Y**2)/(2*(config["initial_wavefunction"]["gaussian_sigma"] / bec.adim_length)**2))

bec.ground_state([trap, contact], callbacks=[], **config["propagation"]["imaginary_time"])



U0 = np.sqrt(8*np.pi)*config["gas"]["N_particles"]*config["potentials"]["contact"]["a_s"]*a_bohr/config["potentials"]["contact"]["a_orth"]*hbar**2/bec.mass
U0_prime = U0/(hbar*bec.adim_pulse*bec.adim_length**2)
mu_prime = np.sqrt(bec.mass*U0*(2*np.pi)**2*config["potentials"]["trap"]["omegax"]*config["potentials"]["trap"]["omegay"]/np.pi)/(hbar*bec.adim_pulse)
V = 2*(np.pi/bec.adim_pulse)**2*((config["potentials"]["trap"]["omegax"]*bec.X)**2+(config["potentials"]["trap"]["omegay"]*bec.Y)**2)
tf_density = (mu_prime-V)/U0_prime * torch.heaviside(mu_prime-V, torch.zeros_like(bec.X))



fig, ax = plt.subplots(1, 1, figsize=(7,4))
ax.plot(bec.x.cpu(), tf_density[bec.psi.shape[0]//2,:].cpu(), label="Thomas-Fermi", ls="dashed", c="red")
ax.plot(bec.x.cpu(), bec.density[bec.psi.shape[0]//2,:].cpu(), label="GPE")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
plt.legend()
plt.title(datetime.now().strftime("%d %b %Y %H:%M:%S"))
fig.savefig("thomas_fermi_profile.png")


