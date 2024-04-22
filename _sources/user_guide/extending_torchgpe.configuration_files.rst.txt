Using configuration files
-------------------------

Sometimes it is useful to define the parameters for the simulations in a separate file. In this case, we suggest to use YAML configuration files, which allow for the definition of lists and dictionaries. Moreover, a custom parser is included in the GPE package to also allow for the evaluation of simple mathematical expressions.

Once a configuration file is defined and stored in the ``path.yaml`` file, its contents can be parsed using the :py:meth:`~torchgpe.utils.configuration.parse_config` function.

.. code-block:: python
    :linenos:

    from torchgpe.utils import parse_config

    data = parse_config('path.yaml')

In the following, we are going to give a few examples for a quick start using configuration files.


Data types
==========

Lists can be defined in a single line as ``[first_element, second_element, third_element]`` or by writing each element in a different line, as in the example below:

.. code-block:: yaml
    :linenos:

    - first_element
    - second_element
    - third_element

.. note::
    The space after the ``-`` character is mandatory. 
    
Regardless of which one of the two notations you decide to use, the parser will output a python list.

Analogously, dictionaries can be defined in a single line as ``{first_key: first_element, second_key: second_element, third_key: third_element}`` or by writing each element in a different line, as in the example below:

.. code-block:: yaml
    :linenos:

    first_key: first_element
    second_key: second_element
    third_key: third_element


.. note::
    The space after the ``:`` character is mandatory.
    
Regardless of which one of the two notations you decide to use, the parser will output a python dictionary.

Of course lists and dictionaries can be mixed. For example, a dictionary can be an element of a list and viceversa:

.. code-block:: yaml
    :linenos:

    first_key:
        - a
        - b
        - c
    second_key: second_element
    third_key: 
        - a1
        - b1
        - k1: k2
          k3: k4
            
.. note::
    The indentation level is not important, as long as it is consistent. For example, both the following lists are valid:

    .. code-block:: yaml
        :linenos:

        first_key:
            - a
            - b
            - c
        second_key: 
                    - a1
                    - b1
                    - c1

    Conversely, the following is not a valid list:

    .. code-block:: yaml
        :linenos:

        first_key:
                - a
            - b
                - c
        

The example below shows how to use the parser to input the most common data types:

.. code-block:: yaml 
    :linenos:

    a: 1                # parses to 'a' -> 1 where 'a' is a string and 1 is an integer
    b: 2.0              # parses to 'b' -> 2.0 where 'b' is a string and 2.0 is a float
    c: 1e3              # parses to 'c' -> 1000.0 where 'c' is a string and 1000.0 is a float
    d: e                # parses to 'd' -> 'e' where both 'd' and 'e' are strings
    f: [1, 2.0, g]      # parses to 'f' -> [1, 2.0, 'g'] where 'f' is a string and [1, 2.0, 'g'] is a list
    h:                  # parses to 'h' -> [3, 4.0, 'i'] where 'h' is a string and [3, 4.0, 'i'] is a list
    - 3
    - 4.0
    - i

Variables
=========

Yaml supports the definition of variables:

.. code-block:: yaml
    :linenos:

    # Definition of variables
    j: &name_of_the_variable 780e-9 
    k: *name_of_the_variable 

The first line defines the mapping ``'j' -> 780e-9`` and creates a variable ``name_of_the_variable`` which points to the value ``780e-9``. The second line defines the mapping ``'k' -> 780e-9`` by using the variable ``name_of_the_variable``.

Math expressions
================

In addition to the standard yaml syntax, the parser also supports the evaluation of simple mathematical expressions via the custom ``!eval`` tag. ``!eval`` takes a list of 1 or 2 values, the first being a mathematical expression to be evaluated and the second a dictionary of variables to serve the script. As before, the list can be given in a single line as well as in multiple ones.

.. code-block:: yaml
    :linenos:

    # Evaluates 1+2 and sets l -> 3
    l: !eval 
    - 1+2

    # Eval has access to all the standard operators (+ - * / ** // %) and the square root via sqrt
    m: !eval [ ((((4-(2+1))*10/2)//2)**2)%3+sqrt(2) ] # Evaluates the expression and sets m -> sqrt(2)+1
    
    # Eval has access to the constants pi, c, hbar
    n: !eval [ 2*pi ]

    # Eval can be used to define complex values
    o: !eval [ 1+2j ]

Additional variables can be given to ``!eval`` through a dictionary in the second parameter. For example the variable ``name_of_the_variable`` that we defined earlier:

.. code-block:: yaml
    :linenos:

    p: !eval
        - 2*pi*c/wavelength
        - wavelength: *name_of_the_variable

.. warning:: 
    
    The expression gets evaluated by python. Reserved keywords like ``lambda`` cannot be used in the ``!eval`` expression


The ``!eval`` tag has also access to the :py:func:`~torchgpe.utils.potentials.linear_ramp`, :py:func:`~torchgpe.utils.potentials.quench` and :py:func:`~torchgpe.utils.potentials.s_ramp` functions to allow for the convenient definition of time dependent variables:

.. code-block:: yaml
    :linenos:

    pump_strength: !eval
        - linear_ramp(0, 0, 1, 5e-3) # linear ramp from 0 to 1 in 5 ms
