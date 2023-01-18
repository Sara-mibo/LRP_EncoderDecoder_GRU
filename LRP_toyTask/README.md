```bash
├── model
│   ├── best_checkpoint1
│   └── last_checkpoint1
├── config.yaml
├── generate_data.py
├── lrp_toyTask.ipynb
├── model.py
└── utils_data.py
```

The file `config.yaml` is a configuration file.

The file `generate_data.py` generates data for the toy task.

The folder `model/` contains the trained model for our toy task.

The file `model.py` contains the `class EncoderDecoder` that defines the model.

The file `utils_data.py` provides the functions that are needed for reading data for the toy task.

The notebook `lrp_toyTask.ipynb` is an example of using the LRP function on the toy task model. (It also provides the input relevance attributions which are computed with three other gradient-based explanation methods: Input x Gradient, Integrated Gradients and Saliency.)
