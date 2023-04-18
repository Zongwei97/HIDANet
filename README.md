# HIDANet

This is the official implementation of [HiDAnet: RGB-D Salient Object Detection via Hierarchical Depth Awareness](https://arxiv.org/pdf/2301.07405.pdf), accepted in TIP'23

# Abstract

RGB-D saliency detection aims to fuse multi-modal cues to accurately localize salient regions. Existing works often adopt attention modules for feature modeling, with few methods explicitly leveraging fine-grained details to merge with semantic cues. Thus, despite the auxiliary depth information, it is still challenging for existing models to distinguish objects with similar appearances but at distinct camera distances. In this paper, from a new perspective, we propose a novel Hierarchical Depth Awareness network (HiDAnet) for RGB-D saliency detection. Our motivation comes from the observation that the multigranularity properties of geometric priors correlate well with the neural network hierarchies. To realize multi-modal and multi-level fusion, we first use a granularity-based attention scheme to strengthen the discriminatory power of RGB and depth features separately. Then we introduce a unified cross dual-attention module for multi-modal and multi-level fusion in a coarse-to-fine manner. The encoded multi-modal features are gradually aggregated into a shared decoder. Further, we exploit a multi-scale loss to take full advantage of the hierarchical information. Extensive experiments on challenging benchmark datasets demonstrate that our HiDAnet performs favorably over the state-of-the-art methods by large margins.

# Train and Test

Please follow the train, inference, and evaluation steps:

```
python train.py
python test_produce_maps.py
python test_evaluation_maps.py
```

Make sure that you have changes the path to your dataset in the [config file](https://github.com/Zongwei97/HIDANet/blob/main/Code/utils/options.py)
