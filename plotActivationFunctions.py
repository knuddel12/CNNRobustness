import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.use('TkAgg')

""" This file plots the activation functions. """

x_tanh = np.linspace(-10, 10, 1000)
y_tanh = np.tanh(x_tanh)
y_sigmoid = 1/(1 + np.exp(-x_tanh))
def relu(x):
    return(np.maximum(0,x))

""" sigmoid plot """
with sns.axes_style("whitegrid"):
    g = sns.lineplot(x=x_tanh, y=y_sigmoid)
    sns.despine(left=True, bottom=True)
    #g.set(xlim=(-10, 10))
    g.tick_params(axis='both', which='major', labelsize=10)

""" tanh plot """
plt.figure()
with sns.axes_style("whitegrid"):
    g = sns.lineplot(x=x_tanh, y=np.tanh(x_tanh))
    sns.despine(left=True, bottom=True)
    #g.set(xlim=(-10, 10))
    g.tick_params(axis='both', which='major', labelsize=10)

""" ReLU plot """
plt.figure()
with sns.axes_style("whitegrid"):
    g = sns.lineplot(x=x_tanh, y=relu(x_tanh))
    sns.despine(left=True, bottom=True)
    g.set(ylim=(-1,None))