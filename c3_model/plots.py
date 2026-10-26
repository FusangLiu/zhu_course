"""Small plotting helpers; all scientific values come from the model results."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyArrowPatch, Circle
from .model import FLUX_NAMES


def style():
    families=['DejaVu Sans']
    # Multiple families enable per-glyph fallback: CJK fonts lack some SI
    # superscripts, while DejaVu Sans supplies those scientific unit glyphs.
    plt.rcParams.update({'font.family':families,'font.sans-serif':families,
                         'axes.unicode_minus':False,'figure.dpi':110,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':12,'axes.labelsize':10})


def pathway(ax, rates, *, added=0.):
    """Main carbon skeleton only; ancillary substrates/PR/sugar branches omitted."""
    names=['RuBP','PGA','DPGA','T3P','FBP','HexP','E4P','SBP','S7P','PenP']
    enzymes=['Rubisco','PGK','GAPDH','FBA','FBPase','TK','FBA','SBPase','TK','PRK']
    flux=['v1','v2','v3','v5','v6','v7','v8','v9','v10','v13']
    theta=np.pi/2-np.arange(10)*2*np.pi/10
    positions=np.c_[np.cos(theta),np.sin(theta)]
    maximum=max(abs(rates[FLUX_NAMES.index(k)]) for k in flux) or 1
    for i,(enzyme,key) in enumerate(zip(enzymes,flux)):
        j=(i+1)%10;p=positions[i];q=positions[j]
        value=float(rates[FLUX_NAMES.index(key)])
        start,end=(p,q) if value>=0 else (q,p)
        ax.add_patch(FancyArrowPatch(start,end,connectionstyle='arc3,rad=-.10',
                     arrowstyle='-|>',mutation_scale=13,shrinkA=27,shrinkB=27,
                     linewidth=1.1+3*abs(value)/maximum,color='#248578',alpha=.82))
        mid=(p+q)/2*1.23
        ax.text(*mid,f'{enzyme}\n{value:.2f}',ha='center',va='center',fontsize=8,color='#26514e')
    for name,position in zip(names,positions):
        ax.add_patch(Circle(position,.115,fc='#f3faf8',ec='#b0d9d0',lw=1.3,zorder=3))
        ax.text(*position,name,ha='center',va='center',fontsize=9,fontweight='bold',zorder=4)
    if added:
        ax.add_patch(FancyArrowPatch(positions[3],positions[4],
            connectionstyle='arc3,rad=.65',arrowstyle='-|>',mutation_scale=14,
            shrinkA=29,shrinkB=29,lw=2,color='#c68b30',linestyle='--'))
        ax.text(.13,-.56,f'Added\nreaction\n{added:.2f}',color='#a16c1c',ha='center',fontsize=9)
    ax.text(0,.08,'Calvin cycle',ha='center',va='center',fontsize=15,fontweight='bold',color='#193f3a')
    ax.text(0,-.12,'Carbon skeleton | Final-time flux',ha='center',fontsize=8,color='#5b7770')
    ax.set(xlim=(-1.62,1.62),ylim=(-1.55,1.55),aspect='equal');ax.axis('off')


def flux_comparison(ax, reference, changed):
    # Reserved slots, diagnostic imports and fixed-cofactor reactions are not
    # mislabeled as extra operative network reactions.
    omitted={'reserved4','reserved35','vdhap_in','vgap_in','vpga_in','vatpf','v60'}
    indices=[i for i,k in enumerate(FLUX_NAMES) if k not in omitted]
    y=np.arange(len(indices));a=reference[indices];b=changed[indices]
    ax.barh(y-.18,a,.35,color='#4a8f82',label='Baseline')
    ax.barh(y+.18,b,.35,color='#d3a451',label='Modified')
    ax.set_yticks(y,[FLUX_NAMES[i] for i in indices],fontsize=8)
    ax.invert_yaxis();ax.set_xlabel('Flux / µmol m⁻² s⁻¹');ax.legend(frameon=False,fontsize=9)
    ax.set_title('Reaction and transport fluxes used in the mass balances')
    ax.grid(axis='x',alpha=.15)
