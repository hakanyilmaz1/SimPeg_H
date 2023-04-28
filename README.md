# SimPeg_H
Magnetometer Datas inversion.
Here we try to get a 3D recovered magnetic
susceptibility model with invertion total magnetic intensity (TMI).

We use theses SimPeg examples:

-Sparse Norm Inversion for Total Magnetic Intensity Data on a Tensor Mesh 
-Slicer demo

The pictures are from Emigma 7.8 for the same datas but with theses (17.55.11.jpeg, 17.55.12.jpeg)








Here we invert total magnetic intensity (TMI) data to recover a magnetic
susceptibility model. We formulate the inverse problem as an iteratively
re-weighted least-squares (IRLS) optimization problem. For this tutorial, we
focus on the following:

    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, optimization)
    - Specifying directives for the inversion
    - Setting sparse and blocky norms
    - Plotting the recovered model and data misfit

Although we consider TMI data in this tutorial, the same approach
can be used to invert other types of geophysical data.

Slicer demo
===========

The example demonstrates the `plot_3d_slicer`

- contributed by `@prisae <https://github.com/prisae>`_

Using the inversion result from the example notebook
`plot_laguna_del_maule_inversion.ipynb <http://docs.simpeg.xyz/content/examples/20-published/plot_laguna_del_maule_inversion.html>`_

In the notebook, you have to use :code:`%matplotlib notebook`.

