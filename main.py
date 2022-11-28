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

cluster1_1, cluster1_2  = [], []
cluster2_1 , cluster2_2 = [], []
cluster3_1 ,cluster3_2 = [], []
cluster4_1 ,cluster4_2 = [], []

for i in range(len(y1)):
    if y1[i]==0:
        cluster1_1.append(x1[i][0])
        cluster1_2.append(x1[i][1])
    elif y1[i]==1:
        cluster2_1.append(x1[i][0])
        cluster2_2.append(x1[i][1])
    elif y1[i]==2:
        cluster3_1.append(x1[i][0])
        cluster3_2.append(x1[i][1])

    elif y1[i]==3:
        cluster4_1.append(x1[i][0])
        cluster4_2.append(x1[i][1])
plt.figure()
plt.scatter(cluster1_1,cluster1_2, color = 'r')
plt.scatter(cluster2_1,cluster2_2, color = 'b')
plt.scatter(cluster3_1,cluster3_2, color = 'k')
plt.scatter(cluster4_1,cluster4_2, color = 'c')
plt.show()


# sorted_index = sorted(range(len(x1)), key = lambda y:y1)
# newX1 = [x1_1[i] for i in sorted_index]
# newX2 = [x1_2[i] for i in sorted_index]
# plt.figure()
# plt.scatter(newX1, newX2, c='b')

gmm = GaussianMixture(n_components = 5)
gmm = gmm.fit(x1)

pred = gmm.predict(x1)

cluster1_1, cluster1_2  = [], []
cluster2_1 , cluster2_2 = [], []
cluster3_1 ,cluster3_2 = [], []
cluster4_1 ,cluster4_2 = [], []

for i in range(len(pred)):
    if pred[i]==0:
        cluster1_1.append(x1[i][0])
        cluster1_2.append(x1[i][1])
    elif pred[i]==1:
        cluster2_1.append(x1[i][0])
        cluster2_2.append(x1[i][1])
    elif pred[i]==2:
        cluster3_1.append(x1[i][0])
        cluster3_2.append(x1[i][1])

    elif pred[i]==3:
        cluster4_1.append(x1[i][0])
        cluster4_2.append(x1[i][1])

plt.figure()
plt.scatter(cluster1_1,cluster1_2, color = 'r')
plt.scatter(cluster2_1,cluster2_2, color = 'b')
plt.scatter(cluster3_1,cluster3_2, color = 'k')
plt.scatter(cluster4_1,cluster4_2, color = 'c')
plt.show()

# sorted_index2 = sorted(range(len(x1)), key = lambda y:pred.all())

# predX1 = [x1_1[i] for i in sorted_index2]
# predX2 = [x1_2[i] for i in sorted_index2]
# plt.scatter(newX1[:-2], newX2[:-2],c = 'r')
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.plot_trisurf(x1_1, x1_2, pred)
# ax.plot_trisurf(x1_1, x1_2, y1)
# plt.show()



