import math
import torch
import torch.optim as optim
from src.schedulers import get_noam_scheduler

def set_optimizer(config, e2e, train_loader):
    optimizer = None
    scheduler = None

    # -- computing steps per epoch considering the accumulation gradient
    if config.training_settings['accum_grad'] != 0:
        steps_per_epoch = math.ceil(len(train_loader) / config.training_settings['accum_grad'])
    else:
        steps_per_epoch = len(train_loader)
    print(f"\nTrainLoader length with a batch size of {config.training_settings['batch_size']}: {len(train_loader)} batches")
    print(f"Accumulation Gradient during {config.training_settings['accum_grad']} steps => Simulated Batch Size of {config.training_settings['batch_size'] * max(1, config.training_settings['accum_grad'])} samples")
    print(f"Computed steps per epoch: {steps_per_epoch}")

    ## -- defining optimizer and scheduler
    if config.training_settings['scheduler'] != "noam":
        print(f"Setting {config.training_settings['optimizer']} optimizer with {config.training_settings['scheduler']} scheduler.")
        if config.training_settings['optimizer'] == "adamw":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, e2e.parameters()), config.training_settings['learning_rate'])
        elif config.training_settings['optimizer'] == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, e2e.parameters()), config.training_settings['learning_rate'], betas=(0.9,0.98), eps=10e-09)

    if config.training_settings['scheduler'] == "noam":
        print(f"Setting {config.training_settings['scheduler']} optimizer-scheduler.")
        optimizer = get_noam_scheduler(
            e2e.parameters(),
            config.training_settings['noam_factor'],
            config.encoder_conf["output_size"],
            config.training_settings['warmup_steps'],
        )

    elif config.training_settings['scheduler'] == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=config.training_settings['learning_rate'],
                                                  steps_per_epoch=steps_per_epoch,
                                                  epochs=config.training_settings['epochs'],
                                                  anneal_strategy="linear")
    else:
        raise RuntimeError("The scheduler should be specified as 'noam' or 'onecycle'")

    return optimizer, scheduler

def save_optimizer(args, optimizer, epoch):
    dst_root = os.path.join(args.output_dir, "optimizer")

    os.makedirs(dst_root, exist_ok=True)
    dst_path = os.path.join(dst_root, "optimizer_" + str(epoch).zfill(3) + ".pth")
    print(f"Saving optimizer in {dst_path} ...")
    torch.save(optimizer.state_dict(), dst_path)
