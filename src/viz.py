import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

import numpy as np


def plot_ts_stack(data,scale=0.9, lw=0.4, title=None, labels=None, fontsize = 30, Ts=0.5, nspace = 50):
    """Plot the multivariate time series stacked vertically.

    Parameters
    ----------
    data : ndarray
        time series of shape [time,nodes]
    scale : float
        amplitude scaling factor
    lw : float
        line width
    title : str
        title of the plot
    labels : list of str
        node labels
    """
    data = data - np.mean(data, axis=1, keepdims=True)
    maxrange = np.max(np.max(data, axis=1) - np.min(data, axis=1))
    data /= maxrange

    n_nodes = data.shape[1]
    n_time  = data.shape[0]
    fig, ax = plt.subplots(figsize=(48,0.5*n_nodes))
    for i in range(n_nodes):
        ax.plot(scale*data[:, i] + i, 'k', lw=lw)
    ax.autoscale(enable=True, axis='both', tight=True)
    if title is not None:
        ax.set_title(title,fontsize=fontsize*2)

    if labels is None:
        labels = np.r_[:n_nodes]
    labels_x = np.r_[:data.shape[0]][0:data.shape[0]:nspace]*Ts
    ax.set_yticks(np.r_[:n_nodes])
    ax.set_xticks(np.r_[:data.shape[0]][0:data.shape[0]:nspace])
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.set_yticklabels(labels,fontsize = fontsize)
    ax.set_xticklabels(labels_x,fontsize = fontsize)
    ax.set_xlabel("Time(sec)",fontsize=fontsize)
    ax.set_ylabel("Regions(numbers)",fontsize=fontsize)


def plot_fcd(FCD, window_step, unit="s", title=None, ax=None, labels=None, colorbar=True,font_size=15):
    """
    FCD:            square FCD matrix of size [w,w]
    window_step:    sliding window increment 
    unit:           time unit of the increment
    labels:         clustering labels of size [w]
    """

    t = FCD.shape[0] * window_step 
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    im = ax.matshow(FCD, extent=[0,t,t,0])

    divider = make_axes_locatable(ax)


    if labels is not None:
        lax = divider.append_axes('right', size='8%', pad=0.05)
        lax.pcolormesh(labels[:,np.newaxis], cmap="tab20")
        lax.invert_yaxis()
        lax.set_title("state",fontsize = font_size)
        lax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
        )
        lax.set_title=("cluster")
    if colorbar:
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('$CC[FC(t_1), FC(t_2)]$',fontsize = font_size)

    if title is not None:
        ax.set_title(title,fontsize = font_size)
    ax.set_xlabel("time [%s]" % unit,fontsize = font_size)
    ax.set_ylabel("time [%s]" % unit,fontsize = font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size) 

def phase_plane(
        model, 
        variables=None, 
        npoints=100, 
        state=None, 
        mode=0,
        title=None,
        square_aspect=False,
        faded=False,
        r_kwds=None,
        V_kwds=None,
        ax=None,
        scale_factor = 1,
        scale_v = 1
):
    assert len(model.state_variables) > 1
    if ax is None:
        _, ax = plt.subplots(figsize=(10,8))

    if faded:
        color="lightgray"
        alpha=0.5
    else:
        color=None
        alpha=1

    if variables==None:
        svarx, svary = [0,1]
        variables = model.state_variables[:2]
    else:
        assert len(variables) == 2
        svarx = model.state_variables.index(variables[0])
        svary = model.state_variables.index(variables[1])

    if state is None:
        state = np.array([model.state_variable_range[key].mean() for key in model.state_variables])
        state = state[:,np.newaxis,np.newaxis]

    if title is None:
        title=model.__class__.__name__


    xlim = model.state_variable_range[variables[0]]*scale_factor
    ylim = model.state_variable_range[variables[1]]*scale_v

    x = np.linspace( *xlim, npoints)
    y = np.linspace( *ylim, npoints)

    xx, yy = np.meshgrid(x,y)
    dx = np.zeros([len(y),len(x)])
    dy = np.zeros([len(y),len(x)])


    nocoupling = np.zeros(( model.nvar, 1, model.number_of_modes))
    for i in range(len(y)):
        for j in range(len(x)):
            state[svarx,0,:] = x[j]
            state[svary,0,:] = y[i]
            dstate = model.dfun(
                    state,
                    nocoupling
            )
            dx[i,j], dy[i,j] = dstate[[svarx,svary],0,mode]
    ax.streamplot(x,y,dx,dy,density=2.0,color=color, zorder=1)

    ax.set_xlabel(variables[0],fontsize=15.0)
    ax.set_ylabel(variables[1],fontsize=15.0)

    if r_kwds is None:
        r_kwds = {'colors':'r', 'alpha':alpha}
    if V_kwds is None:
        V_kwds = {'colors':'g', 'alpha':alpha}

    ax.contour(xx,yy,dx, [0], **r_kwds)
    ax.contour(xx,yy,dy, [0], **V_kwds)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if 'colors' in r_kwds:
        r_kwds['color'] = r_kwds.pop('colors')
    if 'colors' in V_kwds:
        V_kwds['color'] = V_kwds.pop('colors')
    if 'linestyles' in r_kwds:
        r_kwds['linestyle'] = r_kwds.pop('linestyles')
    if 'linestyles' in V_kwds:
        V_kwds['linestyle'] = V_kwds.pop('linestyles')
    legend_elements = [
            Line2D([0], [0], **r_kwds, label=f'$\dot {variables[0]}=0$'),
            Line2D([0], [0], **V_kwds, label=f'$\dot {variables[1]}=0$'),
    ]
    ax.legend(handles=legend_elements,loc='upper right')
    ax.set_title(title)
    if square_aspect:
        ax.set_aspect(abs(xlim[0]-xlim[1])/abs(ylim[0]-ylim[1]))

    return ax


def plot_carpet(ts, imshow_kwds=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    if imshow_kwds is None:
        imshow_kwds = {}
    imshow_kwds.setdefault('aspect','auto')
    imshow_kwds.setdefault('interpolation','none')

    im = ax.imshow(ts, **imshow_kwds)

    return ax, im

