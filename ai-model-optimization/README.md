# Model Optimization Utilities

This project focuses on reducing the computational footprint of deep neural networks for edge deployment.

## Techniques Explored
1. **Quantization**: Converting FP32 parameters to INT8.
2. **Pruning**: Removing redundant connections (unstructured) or entire filters (structured).

## Scripts
- `pruning/` contains utilities to prune specific layers of PyTorch models dynamically.
- `quant/` will host static and dynamic quantization pipelines.
