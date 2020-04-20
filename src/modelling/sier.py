"""
Utility file for plotting the hausdorff and correlation fractal dimension for
Sierpinski triangle.
"""
import numpy as np
import matplotlib.pyplot as plt
import utils.plot_helpers as ph
import utils.helpers as helpers

size = 4096
first_row = np.zeros(size, dtype=int)

first_row[int(size/2)-1] = 1

rows = np.zeros((int(size/2),size),dtype=int)
rows[0] = first_row
for i in range(1,int(size/2)):
    rows[i] = (np.roll(rows[i - 1], -1) + rows[i - 1] + np.roll(rows[i-1],1)) % 2
m = int(np.log(size)/np.log(2))
rows = rows[0:2**(m-1),0:2**m]
plt.matshow(rows, cmap = plt.cm.binary)
plt.clf()

box_count = []
p = []
epsilon = []

for lev in range(2, m):
    block_size = 2**lev
    d0 = 0
    d2 = 0
    for j in range(int(size/(2*block_size))):
        for i in range(int(size/block_size)):
            b = rows[j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size]
            d0 += b.any()
            d2 += np.sum(b)**2
    box_count.append(d0)
    p.append(d2)
    epsilon.append(block_size)

d0_params, _ = helpers.compute_least_sq(epsilon, box_count)
d2_params, _ = helpers.compute_least_sq(epsilon, p)

print("d0 = %.3f" % -d0_params[1])
print("d2 = %.3f" % d2_params[1])

ph.fractal_plot(
    "sier",
    "d0_sier",
    epsilon,
    box_count,
    helpers.fit_func,
    d0_params,
    xlabel=r'$\log \epsilon$', ylabel=r'$\log D0$')

ph.fractal_plot(
    "sier",
    "d2_sier",
    epsilon,
    p,
    helpers.fit_func,
    d2_params,
    xlabel=r'$\log \epsilon$', ylabel=r'$\log D2$')
