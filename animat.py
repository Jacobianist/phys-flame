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


def delZero(Z):
    return np.array([[a if np.abs(a)>0.0002 else np.nan for a in line] for line in Z])
def moveIt(Qu, L, Nt, t, movieName=0):
    plt.close('all')
#    plt.style.use('bmh')
#    lim = np.max(np.abs(Qu)) cmap='RdGy_r' , cmap='Reds'
    vmin, vmax = np.min(Qu), np.max(Qu)
    cmap='RdGy_r'
    fig = plt.figure()
    im = plt.imshow(np.flipud(Qu[0]), cmap=cmap, vmin=-0.6*vmax, vmax=.8*vmax, interpolation='bilinear', animated=True, extent=(0,L,0,L))
    ax = plt.gca()
    ax.grid(False)
    plt.colorbar()
    def animate(z):
        im.set_array(np.flipud(Qu[z]))
        plt.title('time: {:.3f}s'.format(t[z]))
        plt.tight_layout()
        return im,
    anim = animation.FuncAnimation(fig, animate, range(0, Nt), interval=10)
# Save results as videofile
    if movieName != 0:
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=60, bitrate=1500)
        print(time.strftime("%H:%M:%S", time.localtime()), 'animation saving...')
        anim.save('{}_{}.mp4'.format(movieName,time.strftime("%Y%m%d-%H%M%S")), writer=writer)
        np.savez_compressed('{}_{}'.format(movieName,time.strftime("%Y%m%d-%H%M%S")), Qu)
# Or just output
    else:
        return anim
if __name__ == '__main__':
    mvName = 0
    print('{mvName}')
#    aniPlot = moveIt(Qu, L, Nt, t, mvName)
#    my_cmap = plt.cm.RdGy_r
#    my_cmap.set_bad(alpha=0)
#    L,Nt = 10, 800
#    t = np.linspace(0, 4, Nt+1)
#    Qu = np.load('fnkppPC_20171205-012158.npz')['arr_0']
#    interpolation='bilinear'
#    lim = np.ceil(np.max(np.abs(Qu)))
#    #plt.imshow(np.flipud(delZero(Qu[550])), alpha=.6, cmap=my_cmap, interpolation=interpolation, vmin=-lim, vmax=lim, extent=(0,L,0,L))
#    #plt.imshow(np.flipud(delZero(Qu[600])), alpha=1, cmap=my_cmap, interpolation=interpolation, vmin=-lim, vmax=lim, extent=(0,L,0,L))
#    plt.imshow(np.flipud(delZero(Qu[650])), alpha=1, cmap=my_cmap, interpolation=interpolation, vmin=-lim, vmax=lim, extent=(0,L,0,L))
#    plt.colorbar()
##    plt.plot(y*0.5,x,':',linewidth=0.4)
#    # #plt.annotate('point', xy=(0.5,8),xytext=(-15, 25), textcoords='offset points',
#    # #            arrowprops=dict(facecolor='black', shrink=0.05),
#    # #            horizontalalignment='right', verticalalignment='bottom')
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.imshow(np.flipud(delZero(Qu[700])), alpha=.8, cmap=my_cmap, interpolation=interpolation, vmin=-lim, vmax=lim, extent=(0,L,0,L))
#    plt.imshow(np.flipud(delZero(Qu[750])), alpha=.7, cmap=my_cmap, interpolation=interpolation, vmin=-lim, vmax=lim, extent=(0,L,0,L))
#    plt.tight_layout()
#    plt.savefig('fnkppPC.png',bbox_inches = 'tight', pad_inches = 0)
# =============================================================================
#    def animate(i):
#        ax1.clear()
#        plt.plot(x, Qu[i,50,:], color='red', label='u')
#        plt.plot(x, Qv[i,50,:], color='blue', label='v')
#        plt.grid(True)
#        plt.ylim([np.min(Qu), np.max(Qu)])
#        plt.xlim([0, L])
#        plt.xlabel('time: {:03f}s'.format(t[i]))
#        plt.yticks()
#        plt.xticks()
#        plt.tight_layout()
#    plt.close('all')
#    fig = plt.figure()
#    ax1 = fig.add_subplot(1, 1, 1)
#    anim = animation.FuncAnimation(fig, animate, range(0, Nt), interval=10)
#    FFMpegWriter = animation.writers['ffmpeg']
#    writer = FFMpegWriter(fps=60, bitrate=1500)
#    anim.save('d1_e002_q003_f2_{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
