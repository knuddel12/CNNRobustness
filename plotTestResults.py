import numpy as np
from sys import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import seaborn as sns
import csv
import os
""" This file plots the results for the Results-Chapter.
    It uses the .txt files created in the test_nets file. """

def getresults(filepath):
    """ Help function to easily extract results of .txt files into two lists"""
    file = np.genfromtxt(os.path.join(base_path, '%s.txt' % (filepath)), dtype='str')
    results = file[1:]
    xrange = file[0][2:].astype(float)
    return results, xrange

base_path = r'testresults'
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rcParams.update({'font.size': 14})

""" Baseline accuracy """
plt.figure()
filename = 'baseline/acc_results_only_baseline_tested_on_gaussian_blur'
results, xrange = getresults(filename)
for result in results:
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=result[2:].astype(np.float64), label=result[0])
        g.set(ylim=(10, 100))
        g.set(xlim=(0, 4))
        g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")


""" Baseline Losses """
plt.figure()
with open('testresults/Baseline/baseline_train_loss.csv', 'r') as f:
    lines_train = f.readlines()
with open('testresults/Baseline/baseline_valid_loss.csv', 'r') as f:
    lines_valid = f.readlines()
data_train = np.zeros((3, 400))
data_valid = np.zeros((3, 400))
# Loop over each line in the CSV file and split it into three values
for i, line in enumerate(lines_train):
    values = line.strip().split(',')
    # Write the values to the numpy array
    for j in range(3):
        data_train[j, i] = float(values[j])
# Loop over each line in the CSV file and split it into three values
for i, line in enumerate(lines_valid):
    values = line.strip().split(',')
    # Write the values to the numpy array
    for j in range(3):
        data_valid[j, i] = float(values[j])
with sns.axes_style("darkgrid"):
    g = sns.lineplot(x=data_train[0], y=data_train[2], label="training loss")
    g = sns.lineplot(x=data_valid[0], y=data_valid[2], label="validation loss")
    g.set(ylim=(-0.5, 6.5))
    g.set(xlim=(0, 400))
    g.set(xlabel="Epoch", ylabel="Loss")


""" Constant blurs.
    Includes: Gauss (Trained) on Gauss (tested), Box on Gauss, Median on Gauss, Motion on Gauss """
" Gauss on Gauss "
plt.figure()
filename = []
filename.append(f'ConstantBlur/acc_results_const_blur0_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_blur1_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_blur2_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_blur3_zehner_tested_on_gaussian_blur')
results = []
xrange = []
labels = ['\u03C3=0', '\u03C3=1', '\u03C3=2', '\u03C3=3']
for i in range(4):
    results.append(getresults(filename[i])[0])
    xrange.append(getresults(filename[i])[1])
# Plot state after 60 epochs, corresponding to entry #6
entries = [6]
# Line colors
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        for k in range(4):
            g = sns.lineplot(x=xrange[k], y=results[k][i - 1][2:].astype(np.float64), label=labels[k])
        g.set(ylim=(15, 95))
        g.set(xlim=(0, 4))
        plt.legend(title='Std dev', loc='lower right')
        g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")


" box on gauss "
plt.figure()
filename = []
filename.append('ConstantBlur/acc_results_const_box_blur3_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_box_blur5_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_box_blur7_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_box_blur9_zehner_tested_on_gaussian_blur')
results = []
xrange = []
labels = ['3x3', '5x5', '7x7', '9x9']
for i in range(4):
    results.append(getresults(filename[i])[0])
    xrange.append(getresults(filename[i])[1])
# List of entries/network states which are to be plotted
entries = [6]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        for k in range(4):
            g = sns.lineplot(x=xrange[k], y=results[k][i - 1][2:].astype(np.float64), label=labels[k])
        g.set(ylim=(15, 95))
        g.set(xlim=(0, 4))
        plt.legend(title='Kernel size', loc='lower right')
        g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")


" median on gauss "
plt.figure()
filename = []
filename.append('ConstantBlur/acc_results_const_median_blur3_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_median_blur5_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_median_blur7_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_median_blur9_zehner_tested_on_gaussian_blur')
results = []
xrange = []
labels = ['3x3', '5x5', '7x7', '9x9']
for i in range(4):
    results.append(getresults(filename[i])[0])
    xrange.append(getresults(filename[i])[1])
# List of entries/network states which are to be plotted
entries = [6]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        for k in range(4):
            g = sns.lineplot(x=xrange[k], y=results[k][i - 1][2:].astype(np.float64), label=labels[k])
        g.set(ylim=(15, 95))
        g.set(xlim=(0, 4))
        plt.legend(title='Kernel size', loc='lower right')
        g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")


" motion on gauss "
plt.figure()
filename = []
filename.append('ConstantBlur/acc_results_const_median_blur3_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_median_blur5_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_median_blur7_zehner_tested_on_gaussian_blur')
filename.append('ConstantBlur/acc_results_const_median_blur9_zehner_tested_on_gaussian_blur')
results = []
xrange = []
labels = ['3x3', '5x5', '7x7', '9x9']
for i in range(4):
    results.append(getresults(filename[i])[0])
    xrange.append(getresults(filename[i])[1])
# List of entries/network states which are to be plotted
entries = [6]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        for k in range(4):
            g = sns.lineplot(x=xrange[k], y=results[k][i - 1][2:].astype(np.float64), label=labels[k])
        g.set(ylim=(15, 95))
        g.set(xlim=(0, 4))
        plt.legend(title='Kernel size', loc='lower right')
        g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")



""" Accuracy of inbetween-states networks of Constant Blur (Gaussian).
    4 Plots, one for each training sigma """
" Sigma = 3 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_blur3_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 20]
diff_colors = ['#01b908', '#00f767', '#086dea', '#be3bc9', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.axvline(3, 1, 0, color='grey')
g.lines[4].set_linestyle("--")
g.lines[5].set_linestyle("--")
g.lines[5].set_dashes([5, 5])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend(title='Epochs', loc='lower right')

" Sigma = 2 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_blur2_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#01b908', '#00f767', '#00a6e2', '#086dea', '#be3bc9', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.axvline(2, 1, 0, color='grey')
g.lines[5].set_linestyle("--")
g.lines[6].set_linestyle("--")
g.lines[6].set_dashes([5, 5])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend(title='Epochs')

" Sigma = 1 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_blur1_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#01b908', '#00f767', '#00a6e2', '#086dea', '#be3bc9', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.axvline(1, 1, 0, color='grey')
g.lines[5].set_linestyle("--")
g.lines[6].set_linestyle("--")
g.lines[6].set_dashes([5, 5])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
plt.legend(title='Epochs')
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")

" Sigma = 0 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_blur0_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#01b908', '#00f767', '#00a6e2', '#086dea', '#be3bc9', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.axvline(0.02, 1, 0, color='grey')
g.lines[5].set_linestyle("--")
g.lines[6].set_linestyle("--")
g.lines[6].set_dashes([5, 5])
g.set(ylim=(5, 95))
g.set(xlim=(0, 5))
plt.legend(title='Epochs')
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")


""" Step-wise blur change: High to low approach. Gaussian tested on Gaussian"""
plt.figure()
filename = 'Step-wiseBlurChange/results_two_split_gaussian_tested_on_gaussianblur'
results, xrange = getresults(filename)
entries = [1, 2, 3, 4]
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0])
g.set(ylim=(20, 95))
g.set(xlim=(0, 4))
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rcParams.update({'font.size': 14})
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend( loc='lower left')

""" Step-wise blur change: Pre-trained network after blur change to sigma_2 = 0 """
plt.figure()
filename = 'Step-wiseBlurChange/acc_results_continued_3_to_0_tested_on_gaussian_blur'
results, xrange = getresults(filename)
filename_additional = 'ConstantBlur/acc_results_const_blur3_zehner_tested_on_gaussian_blur'
results_additional, xrange_additional = getresults(filename_additional)
with sns.axes_style("darkgrid"):
    g = sns.lineplot(x=xrange_additional, y=results_additional[5][2:].astype(np.float64), label='0', color='#fd8d3c')
    g.set(ylim=(15, 95))
    g.set(xlim=(0, 5))
# List of entries/network states which are to be plotted
entries = [2, 6, 13]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#225ea8','#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0], color=diff_colors[j])
    g.set(ylim=(10, 95))
    g.set(xlim=(0, 5))
    plt.legend(title='Epochs', loc='lower right')
    g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
g.lines[0].set_linestyle("--")

""" Step-wise blur change: Pre-trained network after blur change to sigma_2 = 1.5 """
plt.figure()
filename = 'Step-wiseBlurChange/acc_results_continued_3_to_15_tested_on_gaussian_blur'
results, xrange = getresults(filename)
filename_additional = 'ConstantBlur/acc_results_const_blur3_zehner_tested_on_gaussian_blur'
results_additional, xrange_additional = getresults(filename_additional)
with sns.axes_style("darkgrid"):
    g = sns.lineplot(x=xrange_additional, y=results_additional[5][2:].astype(np.float64), label='0', color='#fd8d3c')
    g.set(ylim=(15, 95))
    g.set(xlim=(0, 5))
# List of entries/network states which are to be plotted
entries = [2, 6, 13]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#225ea8','#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0], color=diff_colors[j])
    g.set(ylim=(10, 95))
    g.set(xlim=(0, 5))
    plt.legend(title='Epochs', loc='lower right')
    g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
g.lines[0].set_linestyle("--")


"""" Step-wise blur change: High to low approach. Gaussian, Box, Median and Motion tested on Gaussian"""
plt.figure()
filename = 'Step-wiseBlurChange/results_all_high_to_low_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 3, 4]
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0])
g.set(ylim=(20, 95))
g.set(xlim=(0, 4))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend( loc='lower left')


""" Epoch-wise blur change: Pre-trained network. Gaussian tested on Gaussian.  """
plt.figure()
filename = 'Epoch-wiseBlurChange/acc_results_continued_iterative_tested_on_gaussian_blur'
results, xrange = getresults(filename)
# Additional file for pre-trained network result
filename_additional = 'ConstantBlur/acc_results_const_blur3_zehner_tested_on_gaussian_blur'
results_additional, xrange_additional = getresults(filename_additional)
with sns.axes_style("darkgrid"):
    g = sns.lineplot(x=xrange_additional, y=results_additional[5][2:].astype(np.float64), label='0', color='#fd8d3c')
# List of entries/network states which are to be plotted
entries = [2, 3, 4, 5, 11]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0], color=diff_colors[j])
    g.set(ylim=(10, 95))
    g.set(xlim=(0, 5))
    plt.legend(title='Epochs', loc='lower right')
    g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
    g.lines[0].set_linestyle("--")


""" Random Blur Training: Gaussian blur tested on Gaussian blur. """
plt.figure()
filename = 'RandomBlur/acc_results_continued_probability_3_to_0_tested_on_gaussian_blur'
results, xrange = getresults(filename)
# Additional file for pre-trained network result
filename_additional = 'ConstantBlur/acc_results_const_blur3_zehner_tested_on_gaussian_blur'
results_additional, xrange_additional = getresults(filename_additional)
with sns.axes_style("darkgrid"):
    g = sns.lineplot(x=xrange_additional, y=results_additional[5][2:].astype(np.float64), label='0', color='#fd8d3c')
    g.set(ylim=(15, 95))
    g.set(xlim=(0, 5))
# List of entries/network states which are to be plotted
entries = [2, 6, 13]
# Line colors
diff_colors = ['#41b6c4', '#225ea8', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0], color=diff_colors[j])
    g.set(ylim=(15, 95))
    g.set(xlim=(0, 5))
    plt.legend(title='Epochs', loc='lower right')
    g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
    g.lines[0].set_linestyle("--")


""" Random Blur Training: Motion blur tested on Gaussian blur. """
plt.figure()
filename = 'RandomBlur/acc_results_continued_motion_probability_9_to_0_tested_on_gaussian_blur'
results, xrange = getresults(filename)
# Additional file for pre-trained network result
filename_additional = 'acc_results_const_blur3_zehner_tested_on_gaussian_blur'
results_additional, xrange_additional = getresults(filename_additional)
with sns.axes_style("darkgrid"):
    g = sns.lineplot(x=xrange_additional, y=results_additional[5][2:].astype(np.float64), label='0', color='#fd8d3c')
    print(results_additional[5][2])
    g.set(ylim=(15, 95))
    g.set(xlim=(0, 5))
# List of entries/network states which are to be plotted
entries = [1, 2, 3, 4, 5, 7]
# Line colors
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0], color=diff_colors[j])
    g.set(ylim=(15, 95))
    g.set(xlim=(0, 5))
    plt.legend(title='Epochs', loc='lower right')
    g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
    g.lines[0].set_linestyle("--")


""" APPENDIX """

""" Baseline tested on box, median and motion """
files = ['Baseline/acc_results_only_baseline_tested_on_box_blur',
         'Baseline/acc_results_only_baseline_tested_on_median_blur',
         'Baseline/acc_results_only_baseline_tested_on_motion_blur']
for filename in files:
    plt.figure()
    results, xrange = getresults(filename)
    entries = [1]
    for i, j in zip(entries, range(len(entries))):
        with sns.axes_style("darkgrid"):
            g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0])
    g.set(ylim=(0, 95))
    g.set(xlim=(1, 13))
    g.set(xlabel="Test image blur / kernel_size'", ylabel="Accuracy / %")
    plt.legend( loc='lower left')
    plt.xticks([1, 3, 5, 7, 9, 11, 13])
    plt.yticks([0, 50, 100])


""" Accuracy of inbetween-states networks of Constant Blur (Motion).
    4 Plots, one for each training sigma """


""" Accuracy of inbetween-states networks of Constant Blur (Motion).
    4 Plots, one for each training kernel size """
" kernel size = 3x3 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_motion_blur3_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend(title='Epochs', loc='lower right')


" kernel size = 5x5 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_motion_blur5_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend(title='Epochs', loc='lower right')


" kernel size = 7x7 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_motion_blur7_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend(title='Epochs', loc='lower right')


" kernel size = 9x9 "
plt.figure()
filename = 'ConstantBlur/acc_results_const_motion_blur9_zehner_tested_on_gaussian_blur'
results, xrange = getresults(filename)
entries = [1, 2, 4, 6, 8, 20]
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#fe0101']
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0],
                         color=diff_colors[j])
g.set(ylim=(10, 95))
g.set(xlim=(0, 5))
g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")
plt.legend(title='Epochs', loc='lower right')



""" Constant blur results: box, median and motion trained networks on box, median, motion test images.
    Opens 9 figures! """
files = ['ConstantBlur/acc_results_all_const_60epochs_tested_on_box_blur',
         'ConstantBlur/acc_results_all_const_60epochs_tested_on_median_blur',
         'ConstantBlur/acc_results_all_const_60epochs_tested_on_motion_blur']
entries = [[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12]]
train_blur_mode = ['motion', 'median', 'box']
test_blur_mode = ['box', 'median', 'motion']
labels = ['3x3', '5x5', '7x7', '9x9']
for file in files:
    results, xrange = getresults(file)
    xrange[0] = 1
    # List of entries/network states which are to be plotted
    # Loop through entries and plot
    for k in range(3):
        plt.figure()
        for i, j in zip(entries[k], range(len(entries[k]))):
            with sns.axes_style("darkgrid"):
                g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=labels[j])
                g.set(ylim=(0, 100))
                g.set(xlim=(1, 13))
                plt.legend(loc='lower right')
                g.set(xlabel="Filter kernel size", ylabel="Accuracy / %")
        plt.xticks([1, 3, 5, 7, 9, 11])
        plt.yticks([0, 50, 100])
""" Constant Gaussian blur trained networks on box, median, motion test images """
files = ['ConstantBlur/acc_results_all_const_60epochs_tested_on_box_blur',
         'ConstantBlur/acc_results_all_const_60epochs_tested_on_median_blur',
         'ConstantBlur/acc_results_all_const_60epochs_tested_on_motion_blur']
entries = [[13, 14, 15, 16]]
train_blur_mode = ['gauss']
test_blur_mode = ['box', 'median', 'motion']
labels = ['\u03C3=0', '\u03C3=1', '\u03C3=2', '\u03C3=3']
for file in files:
    results, xrange = getresults(file)
    xrange[0] = 1
    # List of entries/network states which are to be plotted
    # Loop through entries and plot
    for k in range(1):
        plt.figure()
        for i, j in zip(entries[k], range(len(entries[k]))):
            with sns.axes_style("darkgrid"):
                g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=labels[j])
                g.set(ylim=(0, 100))
                g.set(xlim=(1, 13))
                plt.legend(loc='lower right')
                g.set(xlabel="Test image blur / kernel_size'", ylabel="Accuracy / %")
        plt.xticks([1, 3, 5, 7, 9, 11])
        plt.yticks([0, 50, 100])


""" Step-wise approach, High to low: Gaussian, box, median and motion tested on box, median and motion """
files = ['Step-wiseBlurChange/results_all_low_to_high_tested_on_box_blur',
         'Step-wiseBlurChange/results_all_low_to_high_tested_on_median_blur',
         'Step-wiseBlurChange/results_all_low_to_high_tested_on_motion_blur']
for filename in files:
    plt.figure()
    results, xrange = getresults(filename)
    entries = [1, 2, 3, 4]
    for i, j in zip(entries, range(len(entries))):
        with sns.axes_style("darkgrid"):
            g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0])
    g.set(ylim=(20, 95))
    g.set(xlim=(0, 9))
    g.set(xlabel="Test image blur / kernel_size'", ylabel="Accuracy / %")
    plt.legend( loc='lower left')



""" Not shown in work """

""" Random Gauss blur, sigma=[0,3], p = 0.9 """
filename = 'results_random_blur_gaussian_tested_on_gaussian_blur'
results, xrange = getresults(filename)
# List of entries/network states which are to be plotted
entries = [1]
# Line colors in Hex code
diff_colors = ['#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#e31a1c']
# Loop through entries and plot
for i, j in zip(entries, range(len(entries))):
    with sns.axes_style("darkgrid"):
        g = sns.lineplot(x=xrange, y=results[i - 1][2:].astype(np.float64), label=results[i - 1][0])
        g.set(ylim=(15, 95))
        g.set(xlim=(0, 5))
        g.set(xlabel="Test image blur / \u03C3'", ylabel="Accuracy / %")