import torch
from .configuration import parse_config
import functools
import operator


def ftn(f):
    """Performs an n-dimensional Fourier transform.

    Args:
        f (torch.Tensor): The function to transform.
    """
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(f), norm="ortho"))


def iftn(fk):
    """Performs an n-dimensional inverse Fourier transform.
    
    Args:
        fk (torch.Tensor): The function to anti-transform.
    """
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(fk), norm="ortho"))


def normalize_wavefunction(wavefunction, *d):
    """Normalizes a wavefunction.

    Args:
        wavefunction (torch.Tensor): The wavefunction to normalize.
        *d (float): The grid spacing in all the dimensions.
    """
    return wavefunction / torch.sqrt((torch.abs(wavefunction) ** 2).sum() * functools.reduce(operator.mul, d))


def prompt_yes_no(prompt, default=None):
    """
    Prompts the user with a yes/no question until a valid choice is made.

    Args:
        prompt (str): The prompt to display to the user.
        default (bool, optional): The default value to return if the user does not provide a valid input.

    Returns:
        bool: True if the user responded "y", False if the user responded "n", or the value of the default parameter if specified and the user did not provide a valid input.
    """
    while True:
        response = input(prompt).strip().lower()
        if response == "y":
            return True
        elif response == "n":
            return False
        elif default is not None and response == "":
            return default
        else:
            print('Invalid input. Please enter "y" or "n".')


def enumerate_chunk(l, n):
    """Enumerates a list l in chunks of size n.

    Args:
        l (list): The list to enumerate.
        n (int): The size of each chunk.
    """
    for i in range(0, len(l), n):
        yield zip(range(i, i + n), l[i: i + n])
