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

def save_checkpoint(model, optimizer, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    filename = f'{path}.pth'
    torch.save(checkpoint, filename)

def plot_confusion_matrix(cm, true_labels, pred_labels, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None, threshold=0.5):
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

def test_inference_train(model, criterion, data_loader, device, threshold=0.5):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_samples = 0

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
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(cls_pred.cpu().numpy())  # Now safe, as no gradients are tracked.
    average_loss = total_loss / total_samples
    print('Average Classification Loss:', average_loss)
    return y_true, y_prob, average_loss


def test_inference_test(model, criterion, data_loader, device, threshold=0.5):
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
            y_true.extend(labels.cpu().numpy())
            y_prob[dcm_name[0]] = cls_pred.item()  # .item() safely extracts a Python number
    print('Test Loss (last batch):', loss.item())
    return y_true, np.array(list(y_prob.values())), y_prob

#<-- Previous Draft -->
# import os
# import random
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def stratified_split(csv_df, train_ratio=0.7, test_ratio=0.1, val_ratio=0.2):
#     """
#     Converts the 3‑label “label” column into binary (0: no PP, 1: PP [indeterminate or PP])
#     then splits the data into train, validation, and test sets using stratification.
#     """
#     csv_df['binary_label'] = csv_df['label'].apply(lambda x: 0 if x == 0 else 1)
#     # First split off test (test_ratio)
#     df_temp, df_test = train_test_split(csv_df, test_size=test_ratio, 
#                                         stratify=csv_df['binary_label'], random_state=42)
#     # Then split remaining into train and validation (ratio adjusted accordingly)
#     val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
#     df_train, df_val = train_test_split(df_temp, test_size=val_ratio_adjusted,
#                                          stratify=df_temp['binary_label'], random_state=42)
#     return df_train, df_val, df_test

# def calculate_ci(metric, n, z=1.96):
#     se = np.sqrt((metric * (1 - metric)) / n)
#     return metric - z * se, metric + z * se

# def calculate_auc_ci(y_true, y_probs, n_bootstraps=1000, seed=42, alpha=0.95):
#     rng = np.random.RandomState(seed)
#     bootstrapped_scores = []
#     for _ in range(n_bootstraps):
#         indices = rng.randint(0, len(y_probs), len(y_probs))
#         if len(np.unique(y_true[indices])) < 2:
#             continue
#         score = roc_auc_score(y_true[indices], y_probs[indices])
#         bootstrapped_scores.append(score)
#     sorted_scores = np.sort(bootstrapped_scores)
#     lower_bound = np.percentile(sorted_scores, ((1.0 - alpha) / 2.0) * 100)
#     upper_bound = np.percentile(sorted_scores, (alpha + (1.0 - alpha) / 2.0) * 100)
#     return lower_bound, upper_bound

# def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
#     plt.figure(figsize=(6, 5))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.0
#     for i, j in np.ndindex(cm.shape):
#         plt.text(j, i, format(cm[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()

# def test_inference(model, dataloader, device):
#     model.eval()
#     all_labels = []
#     all_probs = []
#     with torch.no_grad():
#         for imgs, labels in dataloader:
#             imgs = imgs.to(device)
#             outputs = model(imgs)
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_probs.extend(probs.flatten().tolist())
#             all_labels.extend(labels.numpy().tolist())
#     return np.array(all_labels), np.array(all_probs)
