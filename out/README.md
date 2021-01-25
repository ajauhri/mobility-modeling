Synthetic Data
==============

For each city there exists two folders:
* {city_name}_src, having data of pickup/source locations
* {city_name}_des, having data of destination/drop-off locations

Each folder contains 24 files, each having data representing a typical hour of ride requests for the city. One can read the data in python using: 
```
Python 3.6.9 (default, Oct  8 2020, 12:12:24) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> a = np.load('t0.npy')
>>> a.shape
(9168, 2)
```

In the example above file `t0.npy` represents the time between midnight and 1am which would typically having 9k requests.
