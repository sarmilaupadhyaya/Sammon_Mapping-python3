
# application of sammon

import matplotlib.pyplot as plt
import numpy as np
from Sammon_Mapping import SammonMapping

h = np.loadtxt('bouquet_of_circles.txt', dtype= 'f')

import datetime
a = datetime.datetime.now()
sm = SammonMapping(h)
X = sm.get_sammon_mapped_coordinate()
b = datetime.datetime.now()
print(b-a)

x_index = 0
y_index = 1
plt.figure(figsize=(10, 10))
plt.scatter(X[:, x_index], X[:, y_index])

plt.tight_layout()
plt.show()

