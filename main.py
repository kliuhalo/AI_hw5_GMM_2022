from statistics import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from sklearn.mixture import GaussianMixture
import json
import sys


with open('input.json', 'r') as inputFile:
    data = json.load(inputFile)
    x1 = data['x1']
    y1 = data['y1']


    x1_1 = [x1[i][0] for i in range(len(x1))]
    x1_2 = [x1[i][1] for i in range(len(x1))]
print(np.array(x1_1).shape, np.array(x1_2).shape, np.array(y1).shape)


sorted_index = sorted(range(len(x1)), key = lambda y:y1)
newX1 = [x1_1[i] for i in sorted_index]
newX2 = [x1_2[i] for i in sorted_index]
# plt.figure()
# plt.scatter(newX1, newX2, c='b')

gmm = GaussianMixture(n_components = 10)
gmm = gmm.fit(x1)

pred = gmm.predict(x1)

# sorted_index2 = sorted(range(len(x1)), key = lambda y:pred.all())

# print(pred)
# sys.exit()
# predX1 = [x1_1[i] for i in sorted_index2]
# predX2 = [x1_2[i] for i in sorted_index2]
# plt.scatter(newX1[:-2], newX2[:-2],c = 'r')
# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x1_1, x1_2, pred)
ax.plot_trisurf(x1_1, x1_2, y1)
plt.show()



