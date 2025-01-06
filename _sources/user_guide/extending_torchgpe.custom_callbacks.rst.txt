Defining custom callbacks 
-------------------------

Custom callbacks can be defined by subclassing the :class:`~torchgpe.utils.callbacks.Callback` class. In the following example, we will define a callback that monitors where the maximum of the wave function is located and plots it in the xy-plane.

We will use the :class:`DisplacedTrap` implemented in :doc:`extending_torchgpe.custom_potentials` to move the trap center along the custom trajectory

.. math::

    x(\theta) = \sqrt{\theta} \cos(\theta) \, \mu m\\
    y(\theta) = \sqrt{\theta} \sin(\theta) \, \mu m

for :math:`\theta \in \left[0,\,2\pi\right]`.

Before implementing the callback, we can follow the same procedure as in :doc:`extending_torchgpe.custom_potentials`, to see how the trap center moves along the trajectory.

.. image:: ../_static/extending_callback_propagate.gif
    :align: center
    :width: 450
    :alt: Time evolution of the wavefunction in the :class:`DisplacedTrap` potential.


To implement the callback, we first define a class :class:`TrajectoryMonitor` that inherits from :class:`~torchgpe.utils.callbacks.Callback`.

.. code-block:: python
    :linenos:

    from torchgpe.utils.callbacks import Callback

    class TrajectoryMonitor(Callback):
        def __init__(self):
            super().__init__()

All the subclasses of :class:`~torchgpe.utils.callbacks.Callback` are provided with an instance of the :class:`~torchgpe.bec2D.gas.Gas` class at runtime. Therefore, there is no need to pass the :class:`~torchgpe.bec2D.gas.Gas` instance to the constructor of the callback. 

There are four moments in the time evolution of the wave function where the callback can execute custom code, namely before the propagation starts, before and after each step of the split-step Fourier method, and when the propagation finishes. The callback can execute custom code at any of these moments by overriding :py:meth:`~torchgpe.utils.callbacks.Callback.on_propagation_begin`, :py:meth:`~torchgpe.utils.callbacks.Callback.on_epoch_begin`, :py:meth:`~torchgpe.utils.callbacks.Callback.on_epoch_end`, or :py:meth:`~torchgpe.utils.callbacks.Callback.on_propagation_end`.

In this example we will use :py:meth:`~torchgpe.utils.callbacks.Callback.on_propagation_begin` to initialize the arrays that will store the trajectory of the trap center, :py:meth:`~torchgpe.utils.callbacks.Callback.on_epoch_end` to compute the trap center, and :py:meth:`~torchgpe.utils.callbacks.Callback.on_propagation_end` to plot the trajectory.

Before the propagation starts, the callback is provided with the :class:`~torchgpe.bec2D.gas.Gas` instance (stored in the :py:attr:`~torchgpe.utils.callbacks.Callback.gas` attribute), and with important parameters of the propagation (stored in the :py:attr:`~torchgpe.utils.callbacks.Callback.propagation_params` dictionary). 

.. note::
    Currently, only the ``'potentials'``, ``'time_step'`` and ``'N_iterations'`` keys are available for imaginary or real time propagation. In addition, for real time propagation, ``'final_time'`` is also provided.

We use the :py:attr:`N_iterations` parameter to allocate the array that will store the trajectory of the trap center.

.. code-block:: python
    :linenos:

    from torchgpe.utils.callbacks import Callback

    import numpy as np

    class TrajectoryMonitor(Callback):
        def __init__(self):
            super().__init__()

        def on_propagation_begin(self):
            self.center = np.empty((self.propagation_params['N_iterations'], 2))

Once the array is allocated, we can use :py:meth:`~torchgpe.utils.callbacks.Callback.on_epoch_end` to compute the trap center. The callback is provided with the current iteration number, which we can use to store the trap center in the corresponding row of the array.

.. code-block:: python
    :linenos:

    from torchgpe.utils.callbacks import Callback

    import numpy as np

    class TrajectoryMonitor(Callback):
        def __init__(self):
            super().__init__()

        def on_propagation_begin(self):
            self.center = np.empty((self.propagation_params['N_iterations'], 2))

        def on_epoch_begin(self, epoch):
            idx_x, idx_y = np.unravel_index( self.gas.density.argmax().cpu(), self.gas.psi.shape )
            self.center[epoch] = self.gas.y[idx_y].cpu(), self.gas.x[idx_x].cpu()

In the code above, we use ``np.unravel_index`` to convert the index of the maximum of the wave function into the corresponding coordinates in the xy-plane.

.. note:: 
    The wave function of the gas might be stored on the GPU, while numpy arrays are stored on the CPU. Therefore, we need to move the wave function to the CPU before computing the maximum. This is done by calling ``self.gas.density.argmax().cpu()``.

Finally, we use :py:meth:`~torchgpe.utils.callbacks.Callback.on_propagation_end` to plot the trajectory of the trap center.

.. code-block:: python
    :linenos:

    from torchgpe.utils.callbacks import Callback

    import numpy as np
    import matplotlib.pyplot as plt

    class TrajectoryMonitor(Callback):
        def __init__(self):
            super().__init__()

        def on_propagation_begin(self):
            self.center = np.empty((self.propagation_params['N_iterations'], 2))

        def on_epoch_begin(self, epoch):
            idx_x, idx_y = np.unravel_index( self.gas.density.argmax().cpu(), self.gas.psi.shape )
            self.center[epoch] = self.gas.y[idx_y].cpu(), self.gas.x[idx_x].cpu()

        def on_propagation_end(self):

            plt.figure(figsize=(5,5))
            plt.plot(self.center[:,0], self.center[:,1], 'k-')
            plt.xlim(self.gas.x[0], self.gas.x[-1])
            plt.ylim(self.gas.y[0], self.gas.y[-1])
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.savefig('/path/to/folder/trajectory.png')


The picture below, shows the trajectory of the trap center as measured by the callback, on top of the time evolution of the wave function.

.. image:: ../_static/extending_callback_propagate_comparison.gif
    :align: center
    :width: 450
    :alt: Time evolution of the wavefunction in the :class:`DisplacedTrap` potential, and the trajectory of the trap center as measured by the callback.

The callback monitors the trajectory of the trap center and plots it in the xy-plane. The trajectory is consistent with the custom trajectory defined by the :class:`DisplacedTrap` potential, with small deviations due to diabatic effects during the evolution.