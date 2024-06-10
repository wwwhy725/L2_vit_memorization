import tyro

import dataclasses

@dataclasses.dataclass
class Args:
    """Description.
    This should show up in the helptext!"""

    """string"""
    feature_folder: str = 'feature'
    train_feature_filename: str = 'train_feature_cifar_10k.npy'
    gen_feature_filename: str = 'gen_feature_cifar_1024.npy'

    generate_image_path: str = 'img/gen_cifar.npy'
    trainset_path: str = 'img/cifar_subset_10k.npy'

    visualize_memo_path: str = 'visualize/cifar_10k.png'

    pretrain_vit_id: str = 'google/vit-base-patch16-224-in21k'
    
    

    """int number"""
    
    
    """float number"""
    alpha: float = 0.5
    beta: float = 0.5
    threshold: float = 0.15
    

    """bool"""
    


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)