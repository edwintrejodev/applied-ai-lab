# Palm Recognition System

This subsystem handles the extraction of deep features from palm vein imagery and performs cosine similarity matching to authenticate identities.

## Submodules
- `extraction/`: Code to run image inputs through a pre-trained backbone to generate 1D feature vectors.
- `matching/`: Utilities to compute similarity metrics (e.g., Cosine Distance) between feature vectors.

## Dependencies
- torch
- numpy
- opencv-python
