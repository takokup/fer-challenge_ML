import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.utils import get_dataloaders

def execute_epoch(model, dataloader, optimizer, criterion, device, scaler, is_training=True, log_gradients=False):
    if is_training:
        model.train()
    else:
        model.eval()
        
    running_loss, total_correct, total_samples = 0.0, 0, 0
    
    with torch.set_grad_enabled(is_training):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_training:
                optimizer.zero_grad()
                
            # Mixed precision execution matrix for hardware acceleration
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if log_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            wandb.log({f"grad_norm/{name}": param.grad.norm().item()}, commit=False)
                            
                scaler.step(optimizer)
                scaler.update()
                
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
            
    return running_loss / total_samples, total_correct / total_samples

def run_experiment(config, model_class, csv_path, device):
    run = wandb.init(
        project="fer-challenge",
        name=f"{config['model_name']}",
        config=config
    )
    
    train_loader, val_loader = get_dataloaders(
        csv_path, 
        batch_size=config['batch_size'], 
        subset_fraction=config.get('subset_fraction', 1.0),
        use_augmentation=config.get('use_augmentation', True)
    )
    
    model = model_class(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
        
    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
    best_val_acc = 0.0
    log_grads = config.get('log_gradients', False)
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = execute_epoch(model, train_loader, optimizer, criterion, device, scaler, is_training=True, log_gradients=log_grads)
        val_loss, val_acc = execute_epoch(model, val_loader, optimizer, criterion, device, scaler, is_training=False, log_gradients=False)
        
        if scheduler:
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "metrics/generalization_gap_accuracy": train_acc - val_acc,
            "hyperparameters/learning_rate": current_lr
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        if epoch % 5 == 0:
            print(f"[{config['model_name']}] Ep {epoch:2d} -> Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
            
    wandb.run.summary["optimal_validation_accuracy"] = best_val_acc
    run.finish()
