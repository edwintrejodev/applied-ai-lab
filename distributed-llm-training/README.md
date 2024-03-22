# Multi-Node LLM Training Pipeline

An experimental setup for distributed training of large models across multiple GPUs and nodes using PyTorch DDP.

## Overview
This module demonstrates how to wrap a standard PyTorch model with `DistributedDataParallel` and handle distributed samplers to ensure even data distribution.

## Running Tests
To verify the setup:
```bash
pytest tests/
```

## Running DDP
```bash
python core/run_ddp.py --world_size 2
```
