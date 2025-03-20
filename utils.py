import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

def my_seed_everywhere(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_weights_for_epoch(current_epoch, change_epoch, ratio):
    for idx, check_epoch in enumerate(change_epoch):
        if current_epoch < check_epoch:
            return np.array(ratio[idx-1]) / np.sum(ratio[idx-1])
    
    # If current_epoch is greater than all values in change_epoch
    return np.array(ratio[-1]) / np.sum(ratio[-1])

def save_checkpoint(model, optimizer, path):
     # Ensure the parent directory exists.
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created directory: {parent_dir}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    filename = f'{path}.pth'
    try:
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
    
def plot_confusion_matrix(cm, true_labels, pred_labels, threshold, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    plt.figure(figsize=(6, 5))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    tick_marks = np.arange(len(true_labels))
    plt.xticks(tick_marks, pred_labels, rotation=0, fontsize=10)
    plt.yticks(tick_marks, true_labels, rotation=90, fontsize=10, ha='right')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        percent = cm[i, j] / cm[i, :].sum()
        plt.text(j, i-0.1, f"{percent:.2f}", horizontalalignment="center", verticalalignment="center",
                 fontsize=20, color="white" if cm_normalized[i, j] > threshold else "black")
        plt.text(j, i+0.1, f"({cm[i, j]:d})", horizontalalignment="center", verticalalignment="center",
                 fontsize=20, color="white" if cm_normalized[i, j] > threshold else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_ci(metric, n, z=1.96):
    se = np.sqrt((metric * (1 - metric)) / n)
    ci_lower = metric - z * se
    ci_upper = metric + z * se
    return ci_lower, ci_upper

def calculate_auc_ci(y_true, y_probs, seed=42, n_bootstraps=1000, alpha=0.95):
    bootstrapped_aucs = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_probs[indices])
        bootstrapped_aucs.append(score)
    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()
    lower = (1.0 - alpha) / 2.0
    upper = 1.0 - lower
    lower_bound = np.percentile(sorted_scores, lower * 100)
    upper_bound = np.percentile(sorted_scores, upper * 100)
    return lower_bound, upper_bound

def test_inference(model, criterion, data_loader, device, threshold=0.5):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_samples = 0
    results = {}

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels, dcm_name = data['image'], data['label'], data['dcm_name']
            labels = labels.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            cls_pred = torch.sigmoid(logits)
            cls_pred_bin = (cls_pred > threshold).float()
            acc = accuracy_score(labels.cpu().numpy(), cls_pred_bin.cpu().numpy())
            results[dcm_name[0]] = {'Accuracy': acc}
            
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(cls_pred.cpu().numpy())
            
    average_loss = total_loss / total_samples
    
    # all_y_true_np = np.array(y_true)
    # all_y_prob_np = np.array(y_prob) 
    
    print('Average Classification Loss:', average_loss)
    return y_true, y_prob, average_loss, results # all_y_true_np, all_y_prob_np, average_loss, results


## Haven't made edits yet ##
def test_inference_test(model, criterion, data_loader, device, threshold):
    model.eval()
    y_true = []
    y_prob = {}
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels, dcm_name = data['image'], data['label'], data['dcm_name']
            labels = labels.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            cls_pred = torch.sigmoid(logits)
            y_true.extend(labels.detach().numpy())
            y_prob[dcm_name[0]] = cls_pred.item()  # .item() safely extracts a Python number
    print('Test Loss (last batch):', loss.item())
    return y_true, np.array(list(y_prob.values())), y_prob
