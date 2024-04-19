-------------
2D potentials 
-------------

Base classes
============

.. currentmodule:: torchgpe.utils.potentials

.. autoclass:: Potential
    :no-members:

    .. rubric:: Attributes

    .. autosummary::
        :nosignatures:
        :toctree: ../stubs/
        
        ~Potential.is_time_dependent
        ~Potential.gas

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:
        :toctree: ../stubs/

        ~Potential.set_gas
        ~Potential.on_propagation_begin


.. autoclass:: LinearPotential
    :no-members:

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:
        :toctree: ../stubs/

        ~LinearPotential.get_potential


.. autoclass:: NonLinearPotential
    :no-members:

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:
        :toctree: ../stubs/

        ~NonLinearPotential.potential_function


Implemented potentials
======================

2D specific potentials
----------------------

.. currentmodule:: torchgpe.bec2D.potentials

.. autoclass:: Zero
    :no-members:

.. autoclass:: Trap
    :no-members:

.. autoclass:: Contact
    :no-members:

.. autoclass:: Lattice
    :no-members:

.. autoclass:: SquareBox
    :no-members:

.. autoclass:: RoundBox
    :no-members:



.. autoclass:: DispersiveCavity
    :no-members:

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:
        :toctree: ../stubs/

        ~DispersiveCavity.get_alpha
        ~DispersiveCavity.get_order




Time dependent variables
========================

.. currentmodule:: torchgpe.utils.potentials
    
.. autosummary::
    ::no_signatures:
    :toctree: ../stubs/

    ~time_dependent_variable
    ~any_time_dependent_variable
    ~linear_ramp
    ~s_ramp
    ~quench
