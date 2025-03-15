import os
import shutil
import ssl
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, classification_report, confusion_matrix

import config
import utils
import optim
from losses import ClassificationLoss
from model import LateralClassificationModel
from dataset.lateral_dataset_test import LateralDatasetTest
from gradcam import generate_gradcam_visualizations_test

def main():
    args = config.get_args_test()
    ssl._create_default_https_context = ssl._create_unverified_context()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.my_seed_everywhere(args.seed)
    
    run_name = f"{args.weight}_test"
    save_dir = f"./results/{run_name}"
    if os.path.exists(save_dir): 
        shutil.rmtree(save_dir) 
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results directory: {save_dir}")
    
    df = pd.read_csv(args.path)
    # Convert labels: 0 -> 0; 1 and 2 -> 1.
    df['binary_label'] = df['label'].apply(lambda x: 0 if x==0 else 1)
    
    # For external testing, we use the whole CSV.
    test_dataset = LateralDatasetTest(df, args, training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=0, worker_init_fn=utils.seed_worker)
    
    model = LateralClassificationModel(layers=args.layers)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # Load trained weights.
    checkpoint = torch.load(f'./weights/{args.weight}.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = ClassificationLoss().to(DEVICE)
    y_true, y_prob, _ = utils.test_inference_test(model, criterion, test_loader, DEVICE, threshold=args.model_threshold)
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
    
    y_pred = (y_prob >= args.model_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["True Non-PP", "True PP"])
    
    with open(f"{save_dir}/results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nSensitivity: {sen:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    utils.plot_confusion_matrix(cm, ["True Non-PP", "True PP"], ["Pred Non-PP", "Pred PP"],
                                save_path=f"{save_dir}/confusion_matrix.png", threshold=args.model_threshold)
    
    # Generate Grad-CAM visualizations.
    generate_gradcam_visualizations_test(model, args.layers, test_loader, DEVICE, args.model_threshold, save_dir)

if __name__ == "__main__":
    main()
