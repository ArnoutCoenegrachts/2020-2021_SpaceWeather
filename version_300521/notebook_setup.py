## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##
#   FUNCTIONS - Setup
## ---------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------- ##

import sys, os, errno
import tensorflow as tf
import warnings, logging
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.INFO)


## ############################################################################ ##
import sys, os
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#SEED    = 4321
#SEED_tf = 1234
SEED    = 42
SEED_tf = 42

tf.random.set_seed(SEED_tf)
tf.compat.v1.set_random_seed(SEED_tf)

np.random.seed(SEED)


## ############################################################################ ##
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adadelta, Adam
from keras.callbacks import (Callback, TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ProgbarLogger)

import sklearn

import aidapy.aidaxr
from aidapy import load_data

import scripts_f.tools as l_tools

## ############################################################################ ##

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

color_gray   = '#d8dcd6'; 

sns.set_context("notebook")
sns.set_style  ("white") #"whitegrid")

from matplotlib import rcParams
rcParams["savefig.dpi"]        = 100
rcParams["figure.dpi"]         = 100
rcParams["font.size"]          = 16
rcParams["text.usetex"]        = False
rcParams["font.family"]        = ["arial"]
rcParams["font.sans-serif"]    = ["cmss10"]
rcParams["axes.unicode_minus"] = False

mformat_f4="%4.4f" 
mformat_f8="%4.8f"
mformat_e2="%4.2e"



## ############################################################################ ##
association = {0: 'sw', 
               1: 'fs', 
               2: 'bs', 
               3: 'msh', 
               4: 'mp', 
               5: 'bl', 
               6: 'msp', 
               7: 'ps', 
               8: 'psbl', 
               9: 'lobe'}

m_classnames_id = range(10)
m_classnames = ['SW', 'FS', 'BS', 'MSH', 'MP', 'BL', 'MSP', 'PS', 'PSBL', 'Lobe']

def plot_conf_mat (cf_matrix_cnt, cf_matrix, m_classnames, txt_size=11):
    n=7; m_width=17; m_height=7; m_size=txt_size
    t = m_size-2 if m_size>11 else m_size
    
    fig, ax = plt.subplots(num=None, figsize=(m_width, m_height))
    plt.subplots_adjust(wspace=.5, hspace=.5)
    G=gridspec.GridSpec(2,n+2)

    ## Confusion matrix expressed in counts
    ax1 = plt.subplot(G[0,:n-1])
    s=sns.heatmap(cf_matrix_cnt, 
                  #
                  xticklabels=m_classnames,
                  yticklabels=m_classnames, 
                  #
                  annot = np.around(cf_matrix_cnt,2), 
                  lw=0, alpha=.5,
                  fmt="d", 
                  cmap=plt.cm.binary, 
                  linecolor=color_gray, 
                  linewidths=1.,
                  cbar=False, 
                  annot_kws={"size": t}
                 )

    ax1.set_xlabel("Predicted Label",fontweight="bold", fontsize=m_size+2)
    ax1.set_ylabel("True Label", fontweight="bold", fontsize=m_size+2)
    ax1.set_aspect("equal")
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(m_size) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(m_size)

    a=len(m_classnames)+1
    for j in range(a):
        plt.plot(range(a), j*np.ones(a), color="k", linewidth=1.)
        plt.plot(j*np.ones(a), range(a), color="k", linewidth=1.)
    ax1.set_xlim(-.01,a-1+.03); ax1.set_ylim(a-1+.05, -.01)
    m_tot = np.sum(cf_matrix_cnt, axis=1)
    for jj in range(len(m_tot)) :
        posx=1.015; posy=1-(jj+1)/len(m_classnames) +1/len(m_classnames)*1/2
        ax1.text(posx,posy, f"= {m_tot[jj]}",
                 horizontalalignment="center", verticalalignment="top",
                 visible=True,size=m_size, rotation=0.,
                 ha="left", va="center",
                 bbox=dict(boxstyle="round", ec=(1.0,1.0,1.0), fc=(1.0,1.0,1.0),),
                 transform=plt.gca().transAxes,fontweight="normal",style="italic",
                 color="gray", fontsize=m_size, backgroundcolor=None)



    ## Confusion matrix expressed in % (/true_labels initial count)
    ax2 = plt.subplot(G[1,0:n-1])
    ax3 = plt.subplot(G[1,n])
    sns.heatmap(cf_matrix, 
                xticklabels=m_classnames, 
                yticklabels=m_classnames, 
                #
                annot= np.around(cf_matrix,2), 
                lw=0.5, ax = ax2,
                cmap="Blues", 
                linecolor=color_gray, 
                linewidths=1.,#.5
                cbar_ax=ax3, 
                cbar=True, 
                annot_kws={"size": m_size}
               ) 
    ax2.set_xlabel("Predicted Label",fontweight="bold", fontsize=m_size+2)
    ax2.set_ylabel("True Label", fontweight="bold", fontsize=m_size+2)
    ax2.set_aspect("equal")
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(m_size) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(m_size)

    if False:
        m_title_ = "Confusion matrix estimated on TEST_set"
        plt.text(1,1.05, 
                 m_title_,
                 horizontalalignment="center", 
                 verticalalignment="top",
                 visible=True, size=m_size+3, 
                 rotation=0.,ha="center", va="center",
                 bbox=dict(boxstyle="round", ec=(0.9,0.9,0.9), fc=(0.9,0.9,0.9),),
                 transform=plt.gca().transAxes, 
                 fontweight="bold", 
                 fontsize=m_size+3)


    ax1.set_position([0.050, 0.025, 0.4419, 0.6])
    ax2.set_position([0.050, 0.725, 0.4419, 0.6])
    ax3.set_position([0.400, 0.725, 0.0150, 0.6])
    yticklabels = [l for l in ax3.get_yticks() if l!=""]
    formattedList = ["%.2f" % member for member in yticklabels]
    ax3.set_yticklabels(formattedList)
    
    print()