```bash
├── data
│   ├── train_val_test_meanstd_1hotEnc.p
│   ├── train_val_test_minmax_1hotEnc.p
│   └── x_xf_tr_st_NO2.p
├── model
│   ├── config_ozone.yaml
│   └── no_binarystd_GRUh1024l2dr0.3O3v0
│       └── last_checkpoint
├── LRP_example.ipynb
├── model.py
└── utils_data.py
```

The folder `data/` contains pickel files representing mean, standard deviation, min, and max of the data. The file `x_xf_tr_st_NO2.p` contains a couple of samples of data points with high forecast value for NO2 (historical, forecast, and static inputs plus the real measured pollutant values for these points).

The folder `model/` contains the trained pollution forecasting model. It additionally contains a configuration file `config_ozone.yaml`.

The file `model.py` contains the `class EncoderDecoder` that defines the model.

The file `utils_data.py` provides the functions that are needed for reading data for the pollution forecasting task, and loading the configurations.

The notebook `LRP_example.ipynb` is an example of using the LRP function on the pollution forecasting model.
