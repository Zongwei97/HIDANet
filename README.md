# HIDANet

This is the official implementation of [HiDAnet: RGB-D Salient Object Detection via Hierarchical Depth Awareness](https://arxiv.org/pdf/2301.07405.pdf), accepted in TIP'23


# Train and Test

Please follow the train, inference, and evaluation steps:

```
python train.py
python test_produce_maps.py
python test_evaluation_maps.py
```

Make sure that you have changes the path to your dataset in the ![config file](https://github.com/Zongwei97/HIDANet/blob/main/Code/utils/options.py)
