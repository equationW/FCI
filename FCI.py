import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import time
from matplotlib import pyplot as plt



H_total = np.load('HubbardAlpha3N6.npy')
addlist = []
for i in range(84):
    add = 0
    for j in range(84):
        if H_total[i][j] != 0:
            add += 1
    print(add)

plt.subplot(111)
plt.imshow(H_total[:84,84*3:84*4], cmap=plt.get_cmap('PRGn'), vmax=-3, vmin=3)
plt.colorbar(shrink=0.5)
plt.show(block=True)

'''
print(np.shape(H_total[:84,:84]))

start = time.perf_counter()

e, v = linalg.eigh(H_total[:84, :84])
print(v)

elapsed = (time.perf_counter() - start)
print("Time used:", elapsed)


np.save('eA3N6FirstBlock', e)
np.save('vA3N6FirstBlock', v)
'''
