import matplotlib.pyplot as plt
import numpy as np

def plotFigure(i):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Make data
    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)
    X, Y = np.meshgrid(X, Y)
    
    Z1 = np.sqrt((X ** 2) + (Y ** 2))
    Z2 = np.sqrt(((X - 1) ** 2) + ((Y + 1) ** 2))
    
    f1 = (np.sin(4 * Z1) / Z1) + (np.sin(2.5 * Z2) / Z2)
    f2 = 1 - np.sin(5 * Z1) / Z1
    xlen = len(X)
    ylen = len(Y)
    
    # Create an empty array of str with the same shape as the meshgrid, and
    # populate it with 2 colors in a checkerboard pattern
    colortuple = ('y', 'b')
    colors = np.empty(X.shape, dtype=str)
    for y in range(ylen):
        for x in range(xlen):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]
    
    if i == 2:
        surf = ax.plot_surface(X, Y, f2, facecolors=colors, alpha=0.75, linewidth=0)
    else:
        surf = ax.plot_surface(X, Y, f1, facecolors=colors, alpha=0.75, linewidth=0)
    
    plt.show()
    return fig, ax

plotFigure(1)
plotFigure(2)