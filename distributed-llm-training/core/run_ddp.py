import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class DummyLLM(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.fc(self.emb(x))

def setup_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_process():
    dist.destroy_process_group()

def ddp_worker(rank, world_size):
    print(f"Running DDP worker on rank {rank}.")
    setup_process(rank, world_size)
    
    # Create model and move it to the appropriate device
    model = DummyLLM().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Dummy forward pass
    dummy_input = torch.randint(0, 10000, (8, 32)).to(rank)
    output = ddp_model(dummy_input)
    
    print(f"Rank {rank} processed batch, output shape: {output.shape}")
    
    cleanup_process()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    
    print(f"Spawning {args.world_size} processes for DDP...")
    # NOTE: requires GPUs to run device_ids correctly in real scenarios. 
    # For CPU simulation, DDP behaves differently.
    # mp.spawn(ddp_worker, args=(args.world_size,), nprocs=args.world_size, join=True)
