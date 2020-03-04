
## Metane\Propane semiconductor sensor's data

The data set represents real measurements of semiconductor gas sensors.

### Data folder structure:
Date of measurement -> Gas type -> Gas concentration -> Data file.

The data file name `S1_05_100.csv` contains folowwing information:
* `S1` the type of measuring sensor;
* `05` the serial number of this measurement on this day relative to other measurements;
* `100` gas concentration (ppm).

![Image](https://Lcpssm.github.io/data_description/file_tree.png)

### File structure:
* Rows — samples; Columns — features. No indexes and headers;
* 1 Measurement = 2 consecutive rows (temperature and resistance of the gas sensor);
* Duration = 1 minute with a sampling frequency of 10 Hz; 
* The sequence of rows corresponds to the sequence of measurements over time. 

It is strongly recommended to skip the first 5 measurements (10 rows) due to distortion observed when measuring a new type of gas was started.

Data is available for download at the following link:
[GoogleDrive](https://drive.google.com/open?id=1Qb9TXG2SMsmoO41QBvKoCiFFZMWLfhkr)
Some data processing techniques are demonstrated here:
[Ipython](https://github.com/Lcpssm/Lcpssm.github.io/blob/master/data_processing/data%20processing%20demo.ipynb)
