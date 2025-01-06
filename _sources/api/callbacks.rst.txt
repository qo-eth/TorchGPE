-------------
Callbacks 
-------------

Base class
==========

.. currentmodule:: torchgpe.utils.callbacks

.. autoclass:: Callback
    :no-members:

    .. rubric:: Attributes

    .. autosummary::
        ::no_signatures:
        :toctree: ../stubs/

        ~Callback.gas
        ~Callback.propagation_params

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:
        :toctree: ../stubs/

        ~Callback.on_propagation_begin
        ~Callback.on_epoch_begin
        ~Callback.on_epoch_end
        ~Callback.on_propagation_end
    

Implemented callbacks
=====================

Dimensionality dependent callbacks
----------------------------------

.. currentmodule:: torchgpe.utils.callbacks

.. autoclass:: LInfNorm
    :no-members:

    .. rubric:: Attributes

    .. autosummary::
        ::no_signatures:
        :toctree: ../stubs/

        ~LInfNorm.norms


.. autoclass:: L1Norm
    :no-members:

    .. rubric:: Attributes

    .. autosummary::
        ::no_signatures:
        :toctree: ../stubs/

        ~L1Norm.norms

.. autoclass:: L2Norm
    :no-members:

    .. rubric:: Attributes

    .. autosummary::
        ::no_signatures:
        :toctree: ../stubs/

        ~L2Norm.norms

2D specific callbacks
---------------------

.. currentmodule:: torchgpe.bec2D.callbacks

.. autoclass:: CavityMonitor
    :no-members:

    .. rubric:: Attributes

    .. autosummary::
        ::no_signatures:
        :toctree: ../stubs/

        ~CavityMonitor.pump
        ~CavityMonitor.cavity_detuning
        ~CavityMonitor.alpha
        ~CavityMonitor.times

.. autoclass:: Animation
    :no-members:
