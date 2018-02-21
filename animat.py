import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
rcParams['savefig.dpi'] = 300
#import seaborn as sns
# =============================================================================
#['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot',
# 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette',
# 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
# 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster',
# 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid',
# 'seaborn', 'Solarize_Light2', '_classic_test']
#plt.ioff()
#plt.ion()

def moveIt(Qu, L, Nt, t, movieName=0):
    plt.close('all')
#    plt.style.use('bmh')
#    lim = np.max(np.abs(Qu)) cmap='RdGy_r' , cmap='Reds'
    vmin, vmax = np.min(Qu), np.max(Qu)
#    cmap = 'RdGy_r'
    cmap = 'Reds'
    fig = plt.figure()
    im = plt.imshow(np.flipud(Qu[0]), cmap=cmap, vmin=vmin, vmax=0.7*vmax, interpolation='bilinear', animated=True, extent=(0,L,0,L))
    ax = plt.gca()
    ax.grid(False)
    plt.colorbar()
    def animate(z):
        im.set_array(np.flipud(Qu[z]))
        plt.title('time: {:.3f}s'.format(t[z]))
        plt.tight_layout()
        return im,
    anim = animation.FuncAnimation(fig, animate, range(0, Nt, 2), interval=10)
# Save results as videofile
    if movieName != 0:
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=60, bitrate=500)
        print(time.strftime("%H:%M:%S", time.localtime()), 'animation saving...')
        anim.save('{}_{}.mp4'.format(movieName, time.strftime("%Y%m%d-%H%M%S")), writer=writer)
        np.savez_compressed('{}_{}'.format(movieName, time.strftime("%Y%m%d-%H%M%S")), Qu)
# Or just output
    else:
        return anim

def delZero(Z):
    return np.array([[a if np.abs(a)>0.05 else np.nan for a in line] for line in Z])

if __name__ == '__main__':
    plt.close('all')
    # my_cmap = plt.cm.RdGy_r
    # my_cmap = plt.cm.Reds
    # my_cmap.set_bad(alpha=0)
    L, Nx = 20, 200
    dt = 0.0025   
    T = 10           
    Nt = round(T/dt)
    # y = np.ones(Nx)
    # x = np.linspace(0,L,Nx)
    t = np.linspace(0, T, Nt+1)
    dataName = ''
    Q = np.load(dataName+'.npz')['arr_0']
    interpolation='bilinear'
    vmin, vmax = np.min(Q), 0.8*np.max(Q)
    # plt.imshow(np.flipud(delZero(Q[3999])), alpha=1, cmap=my_cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, extent=(0,L,0,L))
    # plt.imshow(np.flipud(delZero(Q[2900])), alpha=1, cmap=my_cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, extent=(0,L,0,L))
    # plt.imshow(np.flipud(delZero(Q[3000])), alpha=1, cmap=my_cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, extent=(0,L,0,L))
    # plt.imshow(np.flipud(delZero(Q[3550])), alpha=1, cmap=my_cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, extent=(0,L,0,L))
    # plt.plot(y*0.5,x,'k:',linewidth=0.4)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('{}_{}.pdf'.format(dataName, time.strftime("%H%M%S")),bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
# =============================================================================
    aniPlot = moveIt(Q, L, Nt, t, 'orkppNC')
