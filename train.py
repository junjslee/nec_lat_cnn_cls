import datetime
import os
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, classification_report, confusion_matrix

import config
import utils
import optim
from losses import ClassificationLoss
from model import LateralClassificationModel
from dataset.lateral_dataset_train import LateralDatasetTrain

def main():
    args = config.get_args_train()
    # ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.my_seed_everywhere(args.seed)

    # Prepare output directory.
    current_time = datetime.datetime.now().strftime("%m%d")
    run_name = f"{current_time}_{args.layers}_lateral_{args.epoch}ep_{args.weight}"
    save_dir = f"./results/{run_name}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results directory: {save_dir}")
    
    # Read CSV and convert labels: 0 remains 0; 1 and 2 become 1.
    df = pd.read_csv(args.path) # pd.read_csv
    # Adjust the DICOM file paths by removing the unwanted prefix.
    df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', ''))
    df['binary_label'] = df['label'].apply(lambda x: 0 if x==0 else 1)
    
    # Stratified split into train (70%), validation (10%), test (20%).
    stratify_col = df['binary_label']
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=stratify_col)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['binary_label'])
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Create datasets.
    train_dataset = LateralDatasetTrain(train_df, args, training=True)
    val_dataset = LateralDatasetTrain(val_df, args, training=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                               num_workers=0, worker_init_fn=utils.seed_worker)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=0, worker_init_fn=utils.seed_worker)
    
    # Build model.
    model = LateralClassificationModel(layers=args.layers)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # Optimizer and scheduler.
    optimizer = optim.create_optimizer(args.optim, model, args.lr_startstep)
    if args.lr_type == 'reduce':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args.lr_patience)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    criterion = ClassificationLoss().to(DEVICE)
    
    best_val_loss = float('inf')
    lrs = []
    losses_train = []
    losses_val = []
    
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}/{args.epoch}")
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels, _ = batch['image'], batch['label'], batch['dcm_name']
            labels = labels.unsqueeze(1)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                return
            loss.backward()
            optimizer.step()
            running_loss += loss_value * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        losses_train.append(epoch_train_loss)
        print(f"Train Loss: {epoch_train_loss:.4f}")
        
        # Validation.
        _, _, epoch_val_loss = utils.test_inference_train(model, criterion, val_loader, DEVICE, threshold=args.model_threshold)
        losses_val.append(epoch_val_loss)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            utils.save_checkpoint(model, optimizer, f'./weights/{run_name}_best')
            print("Model saved!")
            
        scheduler.step(epoch_val_loss)
        lrs.append(optimizer.param_groups[0]["lr"])
    
    # Plot learning curve.
    plt.figure()
    plt.plot(range(1, args.epoch+1), losses_train, label='Train Loss', color='darkred')
    plt.plot(range(1, args.epoch+1), losses_val, label='Val Loss', color='darkblue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()
    
    # After training, load best model and evaluate on the internal test set.
    checkpoint = torch.load(f'./weights/{run_name}_best.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    # For internal test evaluation, use test_df.
    test_dataset = LateralDatasetTrain(test_df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=0, worker_init_fn=utils.seed_worker)
    y_true, y_prob, _ = utils.test_inference_train(model, criterion, test_loader, DEVICE, threshold=args.model_threshold)
    y_true = np.array([item[0] for item in y_true])
    y_prob = np.array(y_prob).flatten()
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ci_lower, ci_upper = utils.calculate_auc_ci(y_true, y_prob, seed=args.seed)
    
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, 
             label=f"ROC curve (area = {roc_auc:.2f}, 95% CI: {ci_lower:.2f}-{ci_upper:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()
    
    # Compute additional metrics.
    y_pred = (y_prob >= args.model_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["True Non-PP", "True PP"])
    print(f"Accuracy: {acc:.4f}   Sensitivity: {sen:.4f}")
    print("Classification Report:\n", report)
    
    with open(f"{save_dir}/results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nSensitivity: {sen:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save confusion matrix.
    utils.plot_confusion_matrix(cm, ["True Non-PP", "True PP"], ["Pred Non-PP", "Pred PP"],
                                save_path=f"{save_dir}/confusion_matrix.png", threshold=args.model_threshold)

    # GradCAM
    generate_gradcam_visualizations_test(model, args.layers, test_loader, DEVICE, args.model_threshold, save_dir)

if __name__ == "__main__":
    main()
