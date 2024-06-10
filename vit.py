from transformers import ViTModel

import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import Args
import tyro

# load configuration
args = tyro.cli(Args)

# device
device = torch.device('cuda') if torch.cuda.is_available() == True else torch.device('cpu')

# resize to 224*224*3
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=None)
    ])

def rescale_to_zero_one(img:torch.Tensor):
    """[-1, 1] to [0, 1]"""
    return (img + 1.) / 2.

# to get the closest factors of a   e.g.  12 = 3 * 4 = 2 * 6  what I want is 3,4 rather than 2,6
def closest_factors(a):
    factors = []
    for i in range(1, int(a ** 0.5) + 1):
        if a % i == 0:
            factors.append((i, a // i))
    closest_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
    return closest_factor

def vit_features(input_imgs:torch.Tensor):
    """input size [b, c, h, w], pixel values in [-1, 1]"""
    b, c, h, w = input_imgs.shape
    assert h == w, "should be square!"
    if h != 224:
        input_imgs = transform(input_imgs)  # resize

    # load vit
    model = ViTModel.from_pretrained(args.pretrain_vit_id).to(device)  # expect input images in [-1, 1]
    model.eval()

    with torch.no_grad():
        outputs = model(input_imgs.to(device))[0]
    features = outputs[:, 0, :]  # [batch, 768]

    return features

def cos_sim(a:torch.Tensor, b:torch.Tensor):
    """input [b, latent_dim]"""
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    return F.cosine_similarity(a[:, None, :], b[None, :, :], dim=-1)

def L2_dist(a:torch.Tensor, b:torch.Tensor):
    """
    input size [b, c, h, w]
    check whether input images are in [0, 1] (so that the output L2_dist can be in [0, 1])
    """

    assert a.min() >= 0 and b.min() >= 0, "images should be in [0, 1]!"

    b1, c1, h1, w1 = a.shape
    b2, c2, h2, w2 = b.shape
    assert (c1, h1, w1) == (c2, h2, w2), "should be of same size except batch size!"

    flatten = c1*h1*w1
    a = a.reshape(b1, flatten)
    b = b.reshape(b2, flatten)

    return torch.sqrt(torch.norm(a[:, None, :] - b[None, :, :], p=2, dim=-1)**2 / flatten)

def visualize_memo(gen:torch.Tensor, train:torch.Tensor, index:torch.Tensor):
    """
    input images should be in [0, 1], a batch of 64 or less is recommended
    index: the topk index of memo_mat
    """
    b, c, h, w = gen.shape
    train_nn = train[index[:, 0]].numpy()
    gen_np = gen.numpy()

    row, col = closest_factors(b)
    canvas = np.zeros((b, h, 2 * w, c))

    for i in range(b):
        cat = np.concatenate((gen_np[i].transpose(1, 2, 0), train_nn[i].transpose(1, 2, 0)), axis=1)
        canvas[i] = cat
    canvas = canvas.reshape(row, col, h, 2*w, c)
    canvas = np.transpose(canvas, axes=(0, 2, 1, 3, 4))
    canvas = canvas.reshape(row*h, col*2*w, c)

    plt.imshow(canvas)
    plt.axis('off')
    plt.savefig(args.visualize_memo_path, dpi=600)
    plt.close()

def memo_dist(gen:torch.Tensor, train:torch.Tensor, feature_folder:str, alpha:float=0.5, beta:float=0.5, threshold:float=0.2):
    """input size [b, c, h, w], pixel values in [-1, 1]"""

    assert gen.min() < 0 and train.min() < 0, "images should be in [-1, 1]!"

    os.makedirs(feature_folder, exist_ok=True)

    # vit features
    try:
        gen_feature_path = os.path.join(feature_folder, args.gen_feature_filename)
        gen_features = vit_features(gen).cpu().detach() 
        np.save(gen_feature_path, gen_features.numpy())
    except Exception as e:
        print(e)
        print("Potential CUDA OOM: Since we assume the batch of generated images is small, \
              we do not implement batched version of generated feature extractor here!")
    
    train_feature_path = os.path.join(feature_folder, args.train_feature_filename)
    if os.path.exists(train_feature_path):
        if not train_feature_path.endswith('.npy'):
            raise ValueError("The file is not in .npy format.")
        else:
            print("features of training data already exist!")
            train_features = np.load(train_feature_path)
            train_features = torch.tensor(train_features)
    else:
        train_features = []
        batch = 1024
        num_batches = (train.shape[0] + batch - 1) // batch
        print("Extracting features from training data...")
        for idx in tqdm(range(num_batches)):
            start_idx = idx * batch
            end_idx = min((idx + 1) * batch, train.shape[0])
            train_features_batch = vit_features(train[start_idx: end_idx]).cpu().detach()
            train_features.append(train_features_batch)
        train_features = torch.concat(train_features, dim=0)
        np.save(train_feature_path, train_features.numpy())

    # L2, remember to rescale images to [0, 1]!
    gen = rescale_to_zero_one(gen)
    train = rescale_to_zero_one(train)
    L2_mat = L2_dist(gen, train)  # smaller is more similar

    # vit cos_sim
    cos_mat = 1. - torch.abs(cos_sim(gen_features, train_features))  # smaller is more similar

    # add with weights
    memo = alpha * L2_mat + beta * cos_mat  # still in [0, 1]
    value, index = torch.topk(memo, k=10, dim=1, largest=False)

    # calculate memo<threshold
    id_memo = torch.nonzero(value < threshold, as_tuple=False)
    memo_num = len(torch.unique(id_memo[:, 0]))
    print(f"memorize {memo_num} images, memorization ratio is {round(memo_num/gen.shape[0], 3)}")

    # visualize memorized images
    visualize_memo(gen[value[:, 0].argsort()], train, index[value[:, 0].argsort()])

# 还差什么呢：train和gen如果batch太大，或许没法处理，要分batch提取特征、计算L2和cos_sim !!!

def main():
    # images
    gen = np.load(args.generate_image_path)
    gen = torch.tensor(gen)
    train = np.load(args.trainset_path)
    train = torch.tensor(train)

    memo_dist(gen, train, feature_folder=args.feature_folder, alpha=args.alpha, beta=args.beta, threshold=args.threshold)

if __name__ == "__main__":
    main()