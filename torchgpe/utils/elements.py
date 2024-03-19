import scipy.constants as spconsts
import numpy as np

# A dictionary of elements.

#: dict(str, dict(str, float)): The dictionary of implemented elements. The keys are the element symbols, and the values are dictionaries of properties. The currently implemnted elements are ``'87Rb'`` and ``'39K'``, while the available properties are: ``m`` (mass), ``omega d2`` (:math:`d_2` line pulse). 
elements_dict = {
    "87Rb": {
        "m": spconsts.physical_constants["atomic mass constant"][0] * 87,
        "omega d2": 2 * np.pi * 384.2304844685e12,
    },
    "39K": {
        "m": spconsts.physical_constants["atomic mass constant"][0] * 39,
        "omega d2": 2 * np.pi * 391.01617003e12,
    },
}

