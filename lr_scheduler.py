# Goal: Control how the learning rate changes during training — start from 0, ramp it up, then slowly bring it back down. Plot that curve.
# Hints:

# The ramp-up phase is called warmup — lr goes from 0 to base_lr linearly over the first warmup_steps
# After warmup, lr decays following a cosine curve back to 0 — look up "cosine annealing formula"
# LambdaLR is a PyTorch tool that lets you control lr with a custom function — look up how to use it
# The function you pass to LambdaLR must return a multiplier (0.0 to 1.0), not the raw lr value
# Plot step (x) vs lr (y) using matplotlib, save to src/lr_schedule.png

import math
from torch.optim.lr_scheduler import LambdaLR
import torch
import matplotlib.pyplot as plt


# def lr_scheduler(total_epoch, current_epoch, base_lr = 1e-3):
#     current_lr = 1e-7
#     epoch_warm_up = total_epoch / 10
#     increment_rate = base_lr / epoch_warm_up
    
#     eta_min = 1e-6
#     decay_epoch_total = total_epoch - epoch_warm_up
    
#     if(current_epoch < epoch_warm_up):
#             current_lr = increment_rate * current_epoch
#     else:
#         current_lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos((current_epoch - epoch_warm_up) * math.pi / decay_epoch_total))

total_epochs = 100
warmup_epochs = 10
base_lr = 1e-3


def lr_scheduler_lambda(current_epoch):

    if(current_epoch < warmup_epochs):
            return current_epoch / warmup_epochs
    
    decay_epochs = total_epochs - warmup_epochs
    epoch_in_decay = current_epoch - warmup_epochs
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * epoch_in_decay / decay_epochs))
    return cosine_decay

def lr_schduler(optimizer, scheduler_lambda):
    return LambdaLR(optimizer, lr_lambda=scheduler_lambda)


if __name__ == "__main__":
    dummy_model_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([dummy_model_param], base_lr)
    scheduler_lambda = lr_scheduler_lambda
    
    scheduler = lr_schduler(optimizer, scheduler_lambda)

    lr_history = []
    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        scheduler.step()

    plt.plot(range(total_epochs), lr_history)
    plt.title("Epoch step VS L.R.")
    plt.xlabel("Epoch Step")
    plt.ylabel("L.R.")
    plt.grid(True)
    
    plt.savefig(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects-CLEAN\lr_scheduler\graph.png")
    plt.show()