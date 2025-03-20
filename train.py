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
import monai
from torch.utils.data._utils.collate import default_collate
from monai.metrics import ConfusionMatrixMetric
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score

import config
import utils
import optim
from gradcam import ModelWrapper, generate_gradcam_visualizations_test
from losses import ClassificationLoss
from model import LateralClassificationModel
from dataset.lateral_dataset import LateralDataset

def train(model, criterion, data_loader, optimizer, device):
    model.train()
    running_loss = 0
    
    for i, data in enumerate(data_loader):
        inputs, labels, _ = data['image'].to(device), data['label'].unsqueeze(1).float().to(device), data['dcm_name']
        
        with torch.set_grad_enabled(True):
            classification_prediction = model(inputs)
            
            # Calculate Loss
            loss, loss_detail = criterion(cls_pred=classification_prediction, cls_gt=labels)
            loss_value = loss.item()
            
            # Loss값이 유한한지 확인
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                return
            if loss_value < 0:
                print("Loss is negative ({}), stopping training.".format(loss_value))
                return
            
            # 역전파 및 옵티마이저를 사용하여 가중치를 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 평균 손실을 계산
            running_loss += loss_value * inputs.size(0) # inputs.size(0) is the batch_size
            
    epoch_loss = running_loss / len(data_loader.dataset) # train 전체 갯수
    print('Train: \n Loss: {:.4f} \n'.format(epoch_loss))
    sample_loss = {'epoch_loss': epoch_loss}
    
    return sample_loss
            
def test(model, criterion, data_loader, device):
    model.eval()   
    
    all_labels = []
    all_preds = []    

    running_loss = 0
    
    confuse_metric = ConfusionMatrixMetric()
    
    for i, data in enumerate(data_loader):
        inputs, labels, _ = data['image'].to(device), data['label'].unsqueeze(1).float().to(device), data['dcm_name']
        
        with torch.set_grad_enabled(True):
            classification_prediction = model(inputs)
            
             # Calculate Loss
            loss, loss_detail = criterion(cls_pred=classification_prediction, cls_gt=labels)
            loss_value = loss.item()
            
            # Loss값이 유한한지 확인
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                return
            
            # 평균 손실을 계산
            running_loss += loss_value * inputs.size(0) # inputs.size(0) is the batch_size
            
            # post-processing
            classification_prediction = torch.sigmoid(classification_prediction) # This is the sigmoid part from NEC_MTL model architecture diagram
    
        all_labels.append(labels.cpu().numpy())
        all_preds.append(classification_prediction.cpu().numpy())
        
        # Metrics CLS
        confuse_metric(y_pred=classification_prediction.round(), y=labels)   # must be rounded
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    # Compute metrics
    f1 = f1_score(all_labels, all_preds.round())
    acc = accuracy_score(all_labels, all_preds.round())
    sen = recall_score(all_labels, all_preds.round()) # Sensitivity is the same as recall
    spe = precision_score(all_labels, all_preds.round(), zero_division=1)
    confuse_metric.reset()
            
    epoch_loss = running_loss / len(data_loader.dataset)
    print('Validation: \n Loss: {:.4f} \n'.format(epoch_loss))
    sample_loss = {'epoch_loss': epoch_loss}
    
    sample_metrics = {'AUC': auc, 'F1': f1, 'Accuracy': acc, 'Sensitivity': sen, 'Specificity': spe}
    print(' AUC: {:.4f} F1: {:.4f} Accuracy: {:.4f} Sensitivity: {:.4f} Specificity: {:.4f} \n'.format(auc, f1, acc, sen, spe))
    
    return sample_loss, sample_metrics

def main():
    args = config.get_args_train()
    # ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.my_seed_everywhere(args.seed)
    
    # Create a unique run name.
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{current_time}_version{args.version}"
    
    # Define output directory
    save_dir = f"/workspace/jun/nec_lat/cnn_classification/results/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Define the weights directory within the mounted path.
    weights_dir = f"/workspace/jun/nec_lat/cnn_classification/results/{run_name}/weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        print(f"Created weights directory: {weights_dir}")
    
    # Define checkpoint path
    checkpoint_path = os.path.join(weights_dir, f"{run_name}_best")
    
    # Read CSV and convert dcm file paths by removing unwanted prefix
    df = pd.read_csv(args.path) # pd.read_csv
    df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', ''))

    # Stratified Key for Pt ID & binary_label
    # df['stratify_key'] = df['Pt ID'].astype(str) + "_" + df['binary_label'].astype(str)
    # Stratified split into train (70%), validation (10%), test (20%).
    # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df['stratify_key'])
    # val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['stratify_key'])
    # Stratified split into train (70%), validation (10%), test (20%).
    stratify_col = df['binary_label']
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=stratify_col)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=args.seed, stratify=temp_df['binary_label'])
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Create datasets.
    train_dataset = LateralDataset(train_df, args, training=True)
    val_dataset = LateralDataset(val_df, args, training=False)
    
    # default_collage == monai.data.utils.default_collate
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, collate_fn=default_collate, shuffle=True, num_workers=0, worker_init_fn=utils.seed_worker)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=default_collate, shuffle=False, num_workers=0, worker_init_fn=utils.seed_worker)
    
    # Build model.
    aux_params=dict(
        pooling='avg',
        dropout=0.5,
        activation=None,
        classes=1,
    )
    model = LateralClassificationModel(layers=args.layers, aux_params=aux_params)
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
    
    # Loss + Necessary Components: weights & epoch
    change_epoch = [0, 100, 120, 135, 160, 170, 175]
    ratio =  [[5, 5], [5, 5],[5, 5],[5, 5], [5, 5],[5, 5],[5, 5]]
    
    lrs = []
    prev_weights = None
    best_loss = float('inf')
    best_auc = 0.0
    # losses_train = []
    # losses_val = []
    
    # losses = {k:     --> from original code
    losses = {k: [] for k in ['train_epoch_loss', 'test_epoch_loss']}
    # metrics = {k:    --> from original code
    metrics = {k: [] for k in ['auc', 'f1', 'acc', 'sen', 'spe']}
    
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}/{args.epoch}\n--------------------------------------------------")
        weights = utils.get_weights_for_epoch(epoch, change_epoch, ratio)
        
        # 이전 가중치와 현재 가중치가 다르면 가중치를 출력합니다.
        if prev_weights is None or not np.array_equal(prev_weights, weights):
            print(f"Weights for Epoch {epoch + 1}: {weights}")
            prev_weights = weights
        
        train_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
        test_criterion = ClassificationLoss(classification_weight=1.0).to(DEVICE)
        
        train_sample_loss = train(model, train_criterion, train_loader, optimizer, DEVICE)
        test_sample_loss, test_sample_metrics = test(model, test_criterion, val_loader, DEVICE)
        
        # 결과를 딕셔너리에 추가
        for key in losses.keys():
            if 'train' in key:
                losses[key].append(train_sample_loss[key.split('train_')[1]])
            else:
                losses[key].append(test_sample_loss[key.split('test_')[1]])
    
        for key in metrics.keys():
            metrics[key].append(test_sample_metrics[key])
            
        # Save best model
        if test_sample_loss['epoch_loss'] < best_loss:
            best_loss = test_sample_loss['epoch_loss']
            utils.save_checkpoint(model, optimizer, checkpoint_path)
            print('Model saved! \n')
        
        scheduler.step(metrics=test_sample_loss['epoch_loss'])
        lrs.append(optimizer.param_groups[0]["lr"])
    
    print("Done!")
    
    #LR
    # Plot and save Learning Rate graph
    plt.plot([i+1 for i in range(len(lrs))], lrs, color='g', label='Learning_Rate')
    plt.savefig(os.path.join(save_dir, "LR.png"))
    # plt.show()

    plt.figure(figsize=(12, 27))

    # First subplot: Total Losses
    plt.subplot(211)
    plt.plot(range(args.epoch), losses['train_epoch_loss'], color='darkred', label='Train Total Loss')
    plt.plot(range(args.epoch), losses['test_epoch_loss'], color='darkblue', label='Val Total Loss')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Total Losses", fontsize=16)
    plt.legend(loc='upper right')

    # Second subplot: F1 Score for Classification
    plt.subplot(212)
    plt.plot(range(args.epoch), metrics['f1'], color='hotpink', label='F1 Score_(CLS)')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("F1 (Classification)", fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "train_val_loss.png"))
    # plt.show()
    plt.close()
    
    # After training, load best model and evaluate on the internal test set.
    checkpoint = torch.load(os.path.join(weights_dir, f"{run_name}_best.pth"), map_location=DEVICE)
    # checkpoint = torch.load(f'/workspace/jun/nec_lat/cnn_classification/weights/{run_name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    test_dataset = LateralDataset(test_df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=default_collate, shuffle=False,
                                              num_workers=0, worker_init_fn=utils.seed_worker)
    
    y_true, y_prob, average_loss, results = utils.test_inference(model, ClassificationLoss(classification_weight=1.0).to(DEVICE), test_loader, DEVICE, threshold=args.model_threshold)

    y_true_flat = [item[0] for item in y_true]
    y_prob_flat = [item[0] for item in y_prob]
    y_prob_flat_rounded = [round(num, 5) for num in y_prob_flat] # 소수점 다섯 자리까지 반올림
    print(f'y_true: \n{y_true_flat}\n')
    print(f'y_prob: \n{y_prob_flat_rounded}\n')
    
    with open(os.path.join(save_dir,f'results_y_true_prob.txt'), 'w', encoding='utf-8') as f:
        f.write(f'y_true: \n{y_true_flat}\n')
        f.write(f'y_prob: \n{y_prob_flat_rounded}\n')

    y_true_np = np.array(y_true_flat)
    y_prob_np = np.array(y_prob_flat_rounded)
    np.savez(f"{save_dir}/results_y_true_prob.npz", y_true=y_true_np, y_prob=y_prob_np)
    
    fpr, tpr, th = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    youden = np.argmax(tpr-fpr)
    ci_lower, ci_upper = utils.calculate_auc_ci(np.array(y_true_flat), np.array(y_prob_flat))
    
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f, 95%% CI: %0.2f-%0.2f)" % (roc_auc, ci_lower, ci_upper))
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(save_dir,f"roc_curve.png"))
    #plt.show()
    plt.close()
    
    thr_val = th[youden]
    y_pred_1 = []
    for prob in y_prob:
        if prob >= thr_val: ### Youden Index를 이용한 classification
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)
    # 이진화 (threshold = 0.5)
    A_pred = (y_prob_np >= thr_val).astype(int)
    # 정확도 (Accuracy) 계산
    accuracy_A = accuracy_score(y_true_np, A_pred)
    # 민감도 (Sensitivity) 계산 (Recall과 동일)
    sensitivity_A = recall_score(y_true_np, A_pred)
    # 특이도 (Specificity) 계산
    # Specificity = TN / (TN + FP)
    conf_matrix_A = confusion_matrix(y_true_np, A_pred)
    specificity_A = conf_matrix_A[0, 0] / (conf_matrix_A[0, 0] + conf_matrix_A[0, 1])
    # Accuracy CI
    ci_accuracy_A = utils.calculate_ci(accuracy_A, len(y_true_np))
    # Sensitivity CI
    ci_sensitivity_A = utils.calculate_ci(sensitivity_A, np.sum(y_true_np == 1))
    # Specificity CI
    ci_specificity_A = utils.calculate_ci(specificity_A, np.sum(y_true_np == 0))
    # 결과 출력
    print(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})")
    print(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})")
    print(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
    print("ROC curve (area = %0.4f, 95%% CI: %0.4f-%0.4f)" % (roc_auc, ci_lower, ci_upper))

    target_names = ["True Non-PP","True PP"]
    report=classification_report(y_true, y_pred_1, target_names=target_names) ### accuracy report
    #결과 저장 + dcm acc, fscore 추가
    with open(os.path.join(save_dir,f'results.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Average Classification Loss: {average_loss}\n')
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Youden index:\n{thr_val}\n')
        # Model Performance Metrics 저장
        f.write(f"\nModel Performance Metrics:\n")
        f.write(f"Model Accuracy: {accuracy_A:.4f}, 95% CI: ({ci_accuracy_A[0]:.4f}, {ci_accuracy_A[1]:.4f})\n")
        f.write(f"Model Sensitivity: {sensitivity_A:.4f}, 95% CI: ({ci_sensitivity_A[0]:.4f}, {ci_sensitivity_A[1]:.4f})\n")
        f.write(f"Model Specificity: {specificity_A:.4f}, 95% CI: ({ci_specificity_A[0]:.4f}, {ci_specificity_A[1]:.4f})\n")
        f.write(f"ROC curve (area = {roc_auc:.2f}, 95% CI: {ci_lower:.4f}-{ci_upper:.4f})\n")
        for dcm_name, metrics in results.items():
            f.write(f'\n dcm_name: {dcm_name}\n')
            for metric, value in metrics.items():
                f.write(f'  {metric}: {value}\n')
                
    cm = confusion_matrix(y_true, y_pred_1)
    utils.plot_confusion_matrix(cm, ["True Non-PP", "True PP"], ["Pred Non-PP", "Pred PP"],save_path=os.path.join(save_dir, f"confusion_matrix.png"))#'/path/to/save/image.png
    print(f'Classification Report:\n{report}')
    print("ROC curve (area = %0.2f)" % auc(fpr, tpr),'\n')
    print( f'weight : \n{run_name}\n' )
    print(f'Thresold Value : {thr_val}')
    
    # Save confusion matrix.
    utils.plot_confusion_matrix(cm, target_names, ["Pred Non-PP", "Pred PP"], threshold=args.model_threshold,
                                save_path=f"{save_dir}/confusion_matrix.png")

    # GradCAM
    modelwrapper = ModelWrapper(model)
    generate_gradcam_visualizations_test(modelwrapper, args.layers, test_loader, DEVICE, args.model_threshold, save_dir)

if __name__ == "__main__":
    main()
