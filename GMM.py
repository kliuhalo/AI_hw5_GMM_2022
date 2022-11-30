from statistics import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from sklearn.mixture import GaussianMixture

from matplotlib.patches import Ellipse

import json
import sys
import argparse

# components = sys.argv[0]
parser = argparse.ArgumentParser(description='n_components:')
parser.add_argument('components', type=int, help='n_components')
args = parser.parse_args()
components = args.components
list2 = ['r','b','k','g','c','y','m']

def show_output(output):
    plt.figure()
    for output in range(components):
        for i in range(len(pred)):
            cluster_x1, cluster_x2 = [], []
            if pred[i] == output:
                cluster_x1.append(x1[i][0])
                cluster_x2.append(x1[i][1])
            plt.scatter(cluster_x1,cluster_x2,color = list2[output])
    plt.show()

def plot_dataset(x1, y1):
    plt.figure()
    for index in range(4):
        cluster_x1, cluster_x2 = [], []
        for i in range(len(y1)):
            if y1[i]==index:
                cluster_x1.append(x1[i][0])
                cluster_x2.append(x1[i][1])
            plt.scatter(cluster_x1,cluster_x2,color = list2[index])
    plt.show()

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))



if __name__=='__main__':
    with open('input.json', 'r') as inputFile:
        data = json.load(inputFile)
        x1 = data['x1']
        y1 = data['y1']
        x1_1 = [x1[i][0] for i in range(len(x1))]
        x1_2 = [x1[i][1] for i in range(len(x1))]
    
    gmm = GaussianMixture(n_components = components)
    gmm = gmm.fit(x1)
    pred = gmm.predict(x1)

    print("gmm means:", gmm.means_)
    print("gmm covariance", gmm.covariances_)
    print("gmm weights", gmm.weights_)

    ax = plt.gca()
    ax.scatter(x1_1, x1_2, c=pred, s=40, cmap='viridis', zorder=2)
    ax.axis('equal')
        
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()

    
    #show_output(components)

