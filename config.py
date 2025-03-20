import argparse

def get_args_train():
    parser = argparse.ArgumentParser(description="Lateral Image Classification Training Arguments")
    # Basic I/O and Paths
    parser.add_argument('--path', type=str, default='workspace/jun/nec_lat/cnn_classification/cnn_cls_prep/lateral_files_extracted.csv', help='Path to CSV with lateral image data')
    parser.add_argument('--gpu', type=str, default='2')
    # parser.add_argument('--external', action='store_true', help='Use external data only (for testing)')
    parser.add_argument('--version', type=int, default=0)
    
    # Model & Training
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--layers', type=str, default='densenet169', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext101_32x4d','resnext101_32x8d','inceptionresnetv2','mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','vgg16','vgg19'])
    parser.add_argument('--batch', type=int, default=6, help='Train batch size.')
    parser.add_argument('--size', type=int, default=1024, help='Target image size.')
    parser.add_argument('--lr_type', type=str, default='reduce', choices=['step', 'reduce'] )
    parser.add_argument('--lr_startstep', type=float, default=0.00005)
    parser.add_argument('--lr_patience', type=int, default=12)
    # parser.add_argument('--half', action='store_true')
    # parser.add_argument('--weight', type=str, default='lateral_densenet169_ep180', help='Identifier for saving weights')
    parser.add_argument('--seed', type=int, default=42)
    
    # Data augmentation parameters
    parser.add_argument('--clahe_cliplimit', type=float, default=2.0)
    parser.add_argument('--clahe_limit', type=int, default=8)    
    parser.add_argument('--clip_min', type=float, default=0.5)
    parser.add_argument('--clip_max', type=float, default=98.5) # 99.5
    parser.add_argument('--rotate_angle', type=float, default=30)
    parser.add_argument('--rotate_percentage', type=float, default=0.8)
    parser.add_argument('--rbc_brightness', type=float, default=0.1) # 0.2
    parser.add_argument('--rbc_contrast', type=float, default=0.2)
    parser.add_argument('--rbc_percentage', type=float, default=0.5)
    parser.add_argument('--elastic_truefalse', action='store_true')
    parser.add_argument('--elastic_alpha', type=float, default=15) # 30
    parser.add_argument('--elastic_sigma', type=float, default=0.75) # 1.5
    parser.add_argument('--elastic_percentage', type=float, default=0.25)
    parser.add_argument('--elastic_alpha_affine', type=float, default=0.45) # 0.9
    parser.add_argument('--gaussian_truefalse', action='store_true')
    parser.add_argument('--gaussian_min', type=float, default=0) # 10.0
    parser.add_argument('--gaussian_max', type=float, default=10) # 50.0
    parser.add_argument('--gaussian_percentage', type=float, default=0.5)
    parser.add_argument('--gamma_truefalse', action='store_true', help='Apply gamma augmentation')
    parser.add_argument('--gamma_min', type=float, default=80.0)
    parser.add_argument('--gamma_max', type=float, default=120.0)
    parser.add_argument('--gamma_percentage', type=float, default=0.5)
    
    # Classification
    # parser.add_argument('--feature', type=str, default='B0')
    parser.add_argument('--model_threshold', type=float, default=0.24387007)
    # parser.add_argument('--model_threshold', type=float, default=0.61940747499)
    # parser.add_argument('--model_threshold_truefalse', action='store_true', help='Fix model threshold to the value provided', default=True)
    #parser.add_argument('--epoch_loss', type=str, default='epoch_loss', choices=['epoch_class_loss', 'epoch_loss'], help='Loss metric for best checkpoint')
    
    return parser.parse_args()

## Haven't made changes yet ##
def get_args_test():
    parser = argparse.ArgumentParser(description="Lateral Image Classification Testing Arguments")
    parser.add_argument('--path', type=str, default='data/lateral_data_test.csv', help='Path to CSV for testing') ###% 경로필요
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--infer', action='store_true', help='Inference mode only')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=125)
    parser.add_argument('--layers', type=str, default='densenet169', choices=['densenet121', 'densenet169','densenet201','densenet161','resnext50_32x4d','se_resnet50','se_resnet101','se_resnext101_32x4d','resnext101_32x8d','inceptionresnetv2','mit_b0','mit_b1','mit_b2','mit_b3','resnet101','resnet152','inceptionv4','mobilenet_v2','resnet50','resnet101','resnext101_32x4d','inceptionv4','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','vgg16','vgg19'])
    parser.add_argument('--batch', type=int, default=6) # 18
    parser.add_argument('--size', type=int, default=1024)
    # parser.add_argument('--lr_type', type=str, default='reduce', choices=['step', 'reduce'])
    # parser.add_argument('--lr_startstep', type=float, default=0.00001)
    # parser.add_argument('--lr_patience', type=int, default=25)
    # parser.add_argument('--half', action='store_true')
    parser.add_argument('--weight', type=str, default='lateral_model', help='Prefix for saved model weights')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature', type=str, default='B0')
    # parser.add_argument('--clahe_cliplimit', type=float, default=2.0)
    parser.add_argument('--model_threshold', type=float, default=0.5)
    return parser.parse_args()
