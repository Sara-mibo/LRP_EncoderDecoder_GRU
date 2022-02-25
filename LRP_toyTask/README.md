```bash
├── config.yaml
├── generate_data.py
├── lrp_toyTask.ipynb
├── model
│   ├── best_checkpoint1
│   └── last_checkpoint1
├── model.py
└── utils_data.py
```

generate_data generates data for the toy task.

Folder model/ contains the trained model for our toy task.

utils_data.py provides the functions that are needed for reading data for the task.

lrp_toyTask.ipynb is an example of using LRP function for the toy task model. It also provides the input attributions which are computed with three other gradient_based methods including Saliency, Input X Gradients, and Integrated Gradients.
