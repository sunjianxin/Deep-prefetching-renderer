# Multi-resolution large-scale volume renderer using RmdnCache

This is a multi-resolution volume rendering framework by inferencing RmdnCache for interactive visualization of large-scale dataset. The detailed results can be visited from the demo video on Youtube from [here](https://youtu.be/SBPq6zV1LUQ).

![result](https://github.com/sunjianxin/Deep-prefetching-renderer/blob/main/teaser.png)


# Dependencies
- CUDA library installed for kernel calls of parallelism.
- TorchLib for making inference on pretrained RmdnCache on CPU
- GLUT for window display and view handling

# Build
```
#cd Deep-prefetching-renderer
#make
```
# Use
```
#sudo ./volumeRender sample_distance algorithm
```
Where sample_distance is the sample distance of neighboring sample points on the ray while executing ray marching algorithm. algorithm is the one of the support cache algorithms with and without prefetching support which are LRU, APPA, Forecache, LSTM and RmdnCache 

# Data set
- Flame dataset: 7GB in size

# Inference

The input is the parameter of the POV of interest and the output is the predicted microblock indices for prefetching. The inferencing is done using TorchLib under C++ environment with optimized performance for interactive visualization.
