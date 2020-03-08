Mobility Modeling
===================

Deep dive in Ride Request Graph (RRG). For more info, please read our paper on [arXiv][1].

To execute: 
```
$ cd mobility-modeling
$ ./src/main.py --input <path_to_configuration_file>
```

A CSV configuration file should have the following coloumns:
* `prefix`: unique identifier for a city like `sf` for San Francisco.
* `start_lat`, `end_lat`, `start_lng`, `end_lng`: define a quadrilateral of the city boundary.
* `time_zone`: Local time zone of the city like `America/Los_Angeles`
* `file_name`: path to a csv file with all ride request data and having the following columns:


A sample configuration file with one entry for the city of San Francisco (SF) would look like:
```
prefix,start_lat,end_lat,start_lng,end_lng,time_zone,file_name
sf,32.8400,34.624491,-118.87654,-117.34905,America/Los_Angeles,san_francisco.csv
```
`file_name` in the configuration where all the data resides should have the following columns:
 * request_timestamp
 * pickup_timestamp
 * pickup_latitude
 * pickup_longitude
 * dropoff_timestamp
 * dropoff_latitude
 * dropoff_longitude

All timestamps are expected in GMT. 

Fractal analysis for ride requests as described here [2] can be done using:
```
$ ./src/main.py --input <path_to_configuration_file> -f -d 
```
We have provided our recommended node lengths for some cities `fractals.py`.


[1]:https://arxiv.org/abs/1701.06635 
[2]:https://arxiv.org/pdf/1712.01235.pdf
