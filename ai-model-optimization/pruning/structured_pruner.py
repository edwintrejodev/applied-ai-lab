import torch
import torch.nn.utils.prune as prune

def prune_conv_layer(module, amount=0.2):
    """
    Applies L1 unstructured pruning to a convolutional layer.
    
    Args:
        module: The PyTorch module (e.g., nn.Conv2d).
        amount: The fraction of weights to prune.
    """
    prune.l1_unstructured(module, name='weight', amount=amount)
    
def remove_pruning_reparametrization(module):
    """
    Makes the pruning permanent by removing the reparametrization.
    """
    try:
        prune.remove(module, 'weight')
    except ValueError:
        pass # Not pruned

if __name__ == '__main__':
    # Demonstration
    import torchvision.models as models
    net = models.resnet18(pretrained=False)
    
    conv1 = net.conv1
    print(f"Weights before pruning: {conv1.weight[0][0][0]}")
    
    prune_conv_layer(conv1, amount=0.3)
    
    print(f"Weights after pruning mask applied: {conv1.weight[0][0][0]}")
    remove_pruning_reparametrization(conv1)
    print("Pruning made permanent.")
