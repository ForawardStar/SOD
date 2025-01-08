## Salient Object Detection via Adaptive Boundary-Location Feature Fusion

Salient object detection aims to identify objects in natural images that most catch human attention. A common approach in this field involves integrating diverse features learned from multiple network branches. Intuitively, to enhance accuracy, the fusion process shall adapt to individual inputs with varying patterns. To improve such adaptability, this project develops a boundary-location information fusion framework. Specifically, we decompose the original problem into two simplified sub-tasks: object localization and boundary refinement, each tackled by a dedicated branch. To mitigate the negative impact of misleading attributions in the feature space, we further propose an adaptive feature augmentation technique, which dynamically pinpoints potential attributions causing incorrect predictions, thereby enhancing robustness. Additionally, an adaptive cross-attention module is devised with parameters that adjust dynamically to effectively aggregate boundary and location features. 

## Environment Installation
Our model is based on Pytorch, which can be installed following the instructions of the official website: https://pytorch.org/

Or, you can install the packages using requirement.txt, through running:
```pip install -r requirement.txt```
