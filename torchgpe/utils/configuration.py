import warnings
import yaml
from math import sqrt
import re
from scipy. constants import pi, hbar, c
from .potentials import linear_ramp, quench, s_ramp

# The global variables that are available to the !eval tag
__globals = {
    # Prevents the user from accessing builtins
    '__builtins__': None,
    # Allows the user to access the sqrt method from the math module
    "sqrt": sqrt,
    # Allows the user to access the linear_ramp, quench, and s_ramp methods from the potentials2D module
    "linear_ramp": linear_ramp,
    "s_ramp": s_ramp,
    "quench": quench,
    # Allows the user to access the pi, hbar, and c constants from the scipy.constants module
    "pi": pi,
    "hbar": hbar,
    "c": c,
}


def __config_tag_evaluate(loader, node):
    """Evaluates a YAML tag of the form !eval <expression> [locals]

    Args:
        loader (yaml.Loader): The YAML loader.
        node (yaml.Node): The YAML node.
    """
    expression = loader.construct_scalar(node.value[0])
    locals = {} if len(
        node.value) == 1 else loader.construct_mapping(node.value[1])

    if any(key in locals for key in __globals.keys()):
        warnings.warn(
            f"{', '.join(__globals.keys())} are reserved keywords and are set to the respective constants. By specifying them, their value is overwritten")

    return eval(expression, __globals, locals)


# Regex for parsing exponential numbers
# Taken from https://stackoverflow.com/questions/30458977/how-to-parse-exponential-numbers-with-pyyaml
__config_exponential_resolver =\
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X)


def parse_config(path):
    """Parses a YAML configuration file.

    Args:
        path (str): The path to the configuration file.

    Returns:
        dict: The parsed configuration.
    
    Raises:
        yaml.YAMLError: If the configuration file is not valid YAML.
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float', __config_exponential_resolver, list(u'-+0123456789.'))
    loader.add_constructor('!eval', __config_tag_evaluate)
    with open(path, "r") as file:
        return yaml.load(file, Loader=loader)
