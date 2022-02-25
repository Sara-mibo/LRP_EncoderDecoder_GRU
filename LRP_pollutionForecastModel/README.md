```bash
├── data
│   ├── train_val_test_meanstd_1hotEnc.p
│   ├── train_val_test_minmax_1hotEnc.p
│   └── x_xf_tr_st_NO2.p
├── LRP_example.ipynb
├── model
│   ├── config_ozone.yaml
│   └── no_binarystd_GRUh1024l2dr0.3O3v0
│       └── last_checkpoint
├── model.py
└── utils_data.py
```

Folder data/ contains pickel files representing mean, standard deviation, min, and max of the data. In x_xf_tr_st_NO2.p, there are a couple of samples of data points with high forecast value for NO2 (historical, forecast, and static inputs plus the real measured values for the points).

Folder model/ contains the trained pollution forecasting model.

utils_data.py provides the functions that are needed for reading data for pollution forecasting task.

LRP_example.ipynb is an example of using LRP function for the pollution forecasting model.
