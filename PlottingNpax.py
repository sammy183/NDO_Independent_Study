# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 22:22:50 2025

Plotting the optimization for fixed npax

@author: NASSAS
"""
from RunOpt import RunDBFOpt
import numpy as np
import multiprocessing
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

prop2 = '18x12E'
prop3 = '22x12E'
def solver_num(solver_name):
    if solver_name == 'APOPT':
        return(1)
    elif solver_name == 'IPOPT':
        return(3)


#%% Generating a plot for a variety of fixed npax for both IPOPT and APOPT
def simple_proc_npax(npax):
    try:
        out = RunDBFOpt(prop2, prop3, fixnpax = npax,
                  folder = False, verbose = False, printout = False, 
                  solver = solver_num('APOPT'))
    except:
        out = np.zeros(18)*np.nan
    return(out)

def simple_totscore_npax_data(nhigh):
    # npaxs = np.linspace(3, 150, n).round(0).astype(int)
    # n = nhigh - 3 + 1
    # npaxs = np.linspace(3, 140, n).round(0).astype(int)
    # npaxs = np.delete(npaxs, np.array([56, 57, 58, 59, 60, 61, 62, 63, 64])) ### NOTE: 61 passengers + APOPT breaks gekko. Why? No clue.
    # see: https://stackoverflow.com/questions/60649331/python-gekko-cant-find-options-json-file
    # also delete 62, 63 pax bc they fuck up somehow  
    
    npaxs = np.concatenate((np.array(list(range(3, 30, 2))), np.array(list(range(70, 140, 2)))))
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        process_func = partial(simple_proc_npax)
        results = list(tqdm(pool.imap(process_func, npaxs)))
        
    dataout = np.array([r for r in results])
    return(npaxs, dataout)

def plot1simple(npaxs, data):
    fig, ax1 = plt.subplots(figsize = (5, 3.5), dpi = 1000) 
    plt.rc('font',family='Arial')
    totscore = data[:, 3]
    M2score = data[:, 0] - 1.0
    M3score = data[:, 1] - 2.0 
    GMscore = data[:, 2]
    ax2 = ax1.twinx()
    ax1.plot(npaxs[totscore > 0.0], totscore[totscore > 0.0], color = '#cc0000', label = 'Total Score', zorder = 1)
    labels = ['M2', 'M3', 'GM']
    colors = ['#666666', '#0343DF', 'orange']
    for i, score in enumerate([M2score, M3score, GMscore]):
        if i == 1:
            ax2.plot(npaxs[score > 0.0], score[score > 0.0], '-', color = colors[i], label = labels[i], alpha = 0.8, zorder = 1)
        else:
            ax2.plot(npaxs[score > 0.0], score[score > 0.0], '--', color = colors[i], label = labels[i], alpha = 0.6, zorder = 1)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False) 
    ax1.set_xlabel(r'Number of Passengers ($n_{pax}$)', fontsize=12.5)
    ax1.set_ylabel('Total Score', color = '#cc0000', fontsize=12.5)
    ax2.set_ylabel('Variable Mission Scores', color = '#666666', fontsize=12.75)    
    ax2.legend(loc='center right', prop={'size': 9}, framealpha=1.0, fancybox = False, edgecolor='black', bbox_to_anchor=(1.0, 0.3))
    # plt.legend()
    # fig.savefig('ProposalPlot1APOPTn200.png', dpi = 800)
    plt.show()

#%% call within main for multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    npax, data = simple_totscore_npax_data(200)
    plot1simple(npax, data)
