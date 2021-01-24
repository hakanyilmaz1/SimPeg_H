"""
Sparse Norm Inversion for Total Magnetic Intensity Data on a Tensor Mesh
========================================================================

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


"""
# %matplotlib notebook

import pyvista as pv
import discretize
from discretize import TensorMesh
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from SimPEG.potential_fields import magnetics
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import (
    maps,
    data,
    inverse_problem,
    data_misfit,
    regularization,
    optimization,
    directives,
    inversion,
)
# Set a documentation friendly plotting theme
pv.set_plot_theme('document')

#############################################
# Load Data and Plot
# ------------------

topo_filename = "DENEME_TOPO.txt"

data_filename = "DENEME_3.obs"

#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic TMI data. Topography is generally
# defined as an (N, 3) array. TMI data is generally defined with 4 columns:
# x, y, z and data.
#

xyz_topo = np.loadtxt(str(topo_filename))
dobs = np.loadtxt(str(data_filename))

receiver_locations = dobs[:, 0:3]
print (receiver_locations)
dobs = dobs[:, 4]
print('dobs')
print(dobs)
# Plot
fig = plt.figure(figsize=(30, 25))
v_max = np.max(np.abs(dobs))

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
plot2Ddata(
    receiver_locations,
    dobs,
    ax=ax1,
    ncontour=1,
    clim=(-v_max, v_max),
    contourOpts={"cmap": "bwr"},
)
ax1.set_title("TMI Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
)
cbar.set_label("$nT$", rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainty
# ------------------
#
# Inversion with SimPEG requires that we define standard deviation on our data.
# This represents our estimate of the noise in our data. For magnetic inversions,
# a constant floor value is generall applied to all data. For this tutorial, the
# standard deviation on each datum will be 2% of the maximum observed magnetics
# anomaly value.
#

maximum_anomaly = np.max(np.abs(dobs))

std = 0.02 * maximum_anomaly * np.ones(len(dobs))

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for the simulation. Magnetic
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations, the list of field components
# which are to be modeled and the properties of the Earth's field.
#

# Define the component(s) of the field we are inverting as a list. Here we will
# invert total magnetic intensity data.
components = ["tmi"]

# Use the observation locations and components to define the receivers. To
# simulate data, the receivers must be defined as a list.
receiver_list = magnetics.receivers.Point(receiver_locations, components=components)

receiver_list = [receiver_list]

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
inclination = 90
declination = 0
strength = 50000
inducing_field = (strength, inclination, declination)

source_field = magnetics.sources.SourceField(
    receiver_list=receiver_list, parameters=inducing_field
)

# Define the survey
survey = magnetics.survey.Survey(source_field)
print (survey)
#############################################
# Defining the Data
# -----------------
#
# Here is where we define the data that is inverted. The data is defined by
# the survey, the observation values and the standard deviations.
#

data_object = data.Data(survey, dobs=dobs, standard_deviation=std)
print ('dobs')
print(dobs)
print('data_object')
print (data_object)
#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to invert TMI data.
# If desired, we could define an OcTree mesh.
#

dh = 5.0
hx = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 10)]
mesh = TensorMesh([hx, hy, hz], "CCN")
print('mesh')
print(mesh)
########################################################
# Starting/Reference Model and Mapping on Tensor Mesh
# ---------------------------------------------------
#
# Here, we would create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the background is 1e-4 SI.
#

# Define background susceptibility model in SI. Don't make this 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.
background_susceptibility = 1e-4
# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each cell

# Define starting model
starting_model = background_susceptibility * np.ones(nC)

##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the magnetics problem by using the simulation
# class.
#

# Define the problem. Define the cells below topography and the mapping
simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    modelType="susceptibility",
    chiMap=model_map,
    actInd=ind_active,
)
print ('simulation')
print(simulation)

#######################################################################
# Define Inverse Problem
# ----------------------
#
# The inverse problem is defined by 3 things:
#
#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem
#

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)

# Define the regularization (model objective function)
reg = regularization.Sparse(
    mesh,
    indActive=ind_active,
    mapping=model_map,
    mref=starting_model,
    gradientType="total",
    alpha_s=0.1,
    alpha_x=0.1,
    alpha_y=0.1,
    alpha_z=0.1,
)

# Define sparse and blocky norms p, qx, qy, qz
reg.norms = np.c_[0, 0, 0, 0]

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=10, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
print('inv_prob')
print(inv_prob)
#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Here we define any directiveas that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio= 1)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4, max_irls_iterations=30, coolEpsFact=1.5, beta_tol=1e-2,
)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

# The directives are defined as a list.
directives_list = [
    sensitivity_weights,
    starting_beta,
    save_iteration,
    update_IRLS,
    update_jacobi,
]

#####################################################################
# Running the Inversion
# ---------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.
#

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)
print('inv')
print(inv)
# Print target misfit to compare with convergence
# print("Target misfit is " + str(target_misfit.target))

# Run the inversion
recovered_model = inv.run(starting_model)
print('recovered_model')
print(recovered_model)

fichier = open("data.txt", "w")
#fichier =("data2")+str(".obs")
fichier.write(str(recovered_model))
#fichier.write(dpred)
fichier.close()

############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Load the true model (was defined on the whole mesh) and extract only the
# values on active cells.
#true_model = np.loadtxt(str(data_filename))
#true_model = true_model[ind_active]
"""
# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
mesh.plotSlice(
    plotting_map * dobs,
    normal="Y",
    ax=ax1,
    ind=int(mesh.nCy / 2),
    grid=True,
    clim=(np.min(dobs), np.max(dobs)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(dobs), vmax=np.max(dobs))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()
"""
# Plot Recovered Model
fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 0 ), # SLICES !!!!    ind=int(mesh.nCy / 2),
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 1 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 1 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()


fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 2 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 2 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 3 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 3 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 4 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 4 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 5 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 5 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 6 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 6 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 7 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 7 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 8 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 8 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

fig = plt.figure(figsize=(18, 8))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
mesh.plotSlice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1 ,
    ind=int( 9 ), # SLICES !!!!
    grid=True,
    clim=(np.min(recovered_model ), np.max(recovered_model)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 9 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])

norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#

# Predicted data with final recovered model
dpred = inv_prob.dpred

 #numpy.savetxt('myFile.csv', matrix, fmt = '%.1f', delimiter = '\t')

# Observed data | Predicted data | Normalized data misfit
data_array = np.c_[dobs, dpred, (dobs - dpred) / std]

fig = plt.figure(figsize=(17, 4))
plot_title = ["Observed", "Predicted", "Normalized Misfit"]
plot_units = ["nT", "nT", ""]

ax1 = 3 * [None]
ax2 = 3 * [None]
norm = 3 * [None]
cbar = 3 * [None]
cplot = 3 * [None]
v_lim = [np.max(np.abs(dobs)), np.max(np.abs(dobs)), np.max(np.abs(data_array[:, 2]))]

for ii in range(0, 3):

    ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.25, 0.84])
    cplot[ii] = plot2Ddata(
        receiver_list[0].locations,
        data_array[:, ii],
        ax=ax1[ii],
        ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]),
        contourOpts={"cmap": "bwr"},
    )
    ax1[ii].set_title(plot_title[ii])
    ax1[ii].set_xlabel("x (m)")
    ax1[ii].set_ylabel("y (m)")

    ax2[ii] = fig.add_axes([0.33 * ii + 0.27, 0.11, 0.01, 0.84])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plt.show()


"""
Slicer demo
===========

The example demonstrates the `plot_3d_slicer`

- contributed by `@prisae <https://github.com/prisae>`_

Using the inversion result from the example notebook
`plot_laguna_del_maule_inversion.ipynb <http://docs.simpeg.xyz/content/examples/20-published/plot_laguna_del_maule_inversion.html>`_

In the notebook, you have to use :code:`%matplotlib notebook`.
"""



###############################################################################
# Download and load data
# ----------------------
#
# In the following we load the :code:`mesh` and :code:`Lpout` that you would
# get from running the laguna-del-maule inversion notebook.


###############################################################################
# Case 1: Using the intrinsinc functionality
# ------------------------------------------
#
# 1.1 Default options
# ^^^^^^^^^^^^^^^^^^^
print('plan un')
mesh.plot_3d_slicer(recovered_model)

###############################################################################
# 1.2 Create a function to improve plots, labeling after creation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Depending on your data the default option might look a bit odd. The look
# of the figure can be improved by getting its handle and adjust it.
"""
def beautify(title, fig=None):
    # Beautify the 3D Slicer result.

    # Get figure handle if not provided
    if fig is None:
        fig = plt.gcf()

    # Get principal figure axes
    axs = fig.get_children()

    # Set figure title
    fig.suptitle(title, y=.95, va='center')

    # Adjust the y-labels on the first subplot (XY)
    plt.setp(axs[1].yaxis.get_majorticklabels(), rotation=90)
    for label in axs[1].yaxis.get_ticklabels():
        label.set_visible(True)
    for label in axs[1].yaxis.get_ticklabels()[::3]:
        label.set_visible(True)
    axs[1].set_ylabel('Northing (m)')

    # Adjust x- and y-labels on the second subplot (XZ)
    axs[2].set_xticks([357500, 362500, 367500])
    axs[2].set_xlabel('Easting (m)')

    plt.setp(axs[2].yaxis.get_majorticklabels(), rotation=90)
    axs[2].set_yticks([2500, 0, -2500, -5000])
    axs[2].set_yticklabels(['$2.5$', '0.0', '-2.5', '-5.0'])
    axs[2].set_ylabel('Elevation (m)')

    # Adjust x-labels on the third subplot (ZY)
    axs[3].set_xticks([2500, 0, -2500, -5000])
    axs[3].set_xticklabels(['', '0.0', '-2.5', '-5.0'])

    # Adjust colorbar
    axs[4].set_ylabel('Density (g/cc$^3$)')

    # Ensure sufficient margins so nothing is clipped
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
"""

###############################################################################
#
#print('plan deux')
#mesh.plot_3d_slicer(recovered_model)

###############################################################################
# 1.3 Set `xslice`, `yslice`, and `zslice`; transparent region
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The 2nd-4th input arguments are the initial x-, y-, and z-slice location
# (they default to the middle of the volume). The transparency-parameter can
# be used to define transparent regions.
print('plan trois')
mesh.plot_3d_slicer(recovered_model, 0, 0, 0, transparent=[[-0.02, 0.1]])

###############################################################################
# 1.4 Set `clim`, use `pcolor_opts` to show grid lines
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print('plan quatre')
mesh.plot_3d_slicer(
    recovered_model, clim=[-4, 2], pcolor_opts={'edgecolor': 'k', 'linewidth': 0.01}
)

###############################################################################
# 1.5 Use `pcolor_opts` to set `SymLogNorm`, and another `cmap`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print('plan cinq')
mesh.plot_3d_slicer(
    recovered_model, pcolor_opts={'norm': SymLogNorm(linthresh=0.01),'cmap': 'RdBu_r'}
)

###############################################################################
# 1.6 Use :code:`aspect` and :code:`grid`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# By default, :code:`aspect='auto'` and :code:`grid=[2, 2, 1]`. This means that
# the figure is on a 3x3 grid, where the `xy`-slice occupies 2x2 cells of the
# subplot-grid, `xz`-slice 2x1, and the `zy`-silce 1x2. So the
# :code:`grid=[x, y, z]`-parameter takes the number of cells for `x`, `y`, and
# `z`-dimension.
#
# :code:`grid` can be used to improve the probable weired subplot-arrangement
# if :code:`aspect` is anything else than :code:`auto`. However, if you zoom
# then it won't help. Expect the unexpected.
print('plan six')
mesh.plot_3d_slicer(recovered_model, aspect= 'auto', grid=[4, 4, 3],transparent='slider')

###############################################################################
# 1.7 Transparency-slider
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Setting the transparent-parameter to 'slider' will create interactive sliders
# to change which range of values of the data is visible.
#print('plan sept')
#mesh.plot_3d_slicer(recovered_model, transparent='slider')

###############################################################################
# Case 2: Just using the Slicer class
# ------------------------------------------
#
# This way you get the figure-handle, and can do further stuff with the figure.
print('plan huit')
# You have to initialize a figure
fig = plt.figure()

# Then you have to get the tracker from the Slicer
tracker = discretize.View.Slicer(mesh, recovered_model)

# Finally you have to connect the tracker to the figure
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

# Run it through beautify


plt.show()
print('plan FIN')






###################################################
"""
mesh_file = "craig_chile.msh"

model_file = recovered_model

###############################################################################
# Read the mesh and model
reader = PVGeo.ubc.TensorMeshReader()
reader.set_mesh_filename(mesh_file)
reader.add_model_file_name(model_file)
mesh = reader.apply()


###############################################################################
# Use a `PyVista` ``threshold`` filter to remove ``NaN`` data values
mesh.threshold().plot()


###############################################################################
# Or inspect slices of the model
mesh.slice_orthogonal().plot()


###############################################################################
# Or threshold a data range
mesh.threshold([-0.6, -0.3]).plot(clim=[-0.6, 0.3])

#####################################################
"""