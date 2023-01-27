import torch

def train_epoch(adjoints, R_tensor, bias_force, optimizer, n_in, args):
    
    perm_indices = torch.randperm(adjoints.shape[0])

    split_indices = torch.split(perm_indices, args.mini_batch)

    for indices in split_indices:

        optimizer.zero_grad()


        grad_minibatch = adjoints[indices]
        force_input_minibatch = R_tensor[indices]
        force_input_minibatch.requires_grad=True
        
        minibatch_force = bias_force(force_input_minibatch, training=True)

        torch.autograd.backward(minibatch_force, grad_tensors=grad_minibatch, retain_graph=False)

        optimizer.step()

