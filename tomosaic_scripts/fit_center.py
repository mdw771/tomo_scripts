import numpy as np


y = np.array([150,
              416,
              683,
              949,
              1214,
              1480,
              1746,
              2011,
              2277,
              2543,
              2810,
              3076,
              3343,
              3609])

centers = np.array([2311,
                    2313,
                    2316,
                    2318,
                    2320,
                    2323,
                    2324,
                    2325,
                    2328,
                    2329,
                    2332,
                    2336,
                    2337,
                    2339])

a, b = np.polyfit(y, centers, 1)

print('center = slice * {} + {}'.format(a, b))
print((y * a + b).astype('int'))
print(centers)