import matplotlib.pyplot as plt
import matplotlib as mpl

def simple_sample_display(sample_data):
    fig, axs = plt.subplots(len(sample_data.columns),1)
    cmap = mpl.colormaps['plasma']
    # Take colors at regular intervals spanning the colormap.
    for n,s in enumerate(sample_data.columns):
        axs[n].set_ylabel(s, fontstyle="normal", fontsize=8,rotation=45)
        rgba = cmap(1/(n+1))
        axs[n].plot(sample_data[s].values, linewidth=2, color=rgba)
        axs[n].get_xaxis().set_ticks([])
        axs[n].get_yaxis().set_ticks([])
    axs[n].get_yaxis().set_ticks([])
    axs[n].set_xlabel("Timesteps")
    plt.show()