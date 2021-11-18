import sys
import argparse
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from backbones import get_model
import torch
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def main(args):
    labels = []
    distances = []

    transform = get_transform()
    net = load_model(args.network, args.weight)

    for label in os.listdir(args.input):
        pairs_path = os.path.join(args.input, label)
        for pair in tqdm(os.listdir(pairs_path)):
            labels.append(label == 'positive')
            images_path = os.path.join(pairs_path, pair)
            images = os.listdir(images_path)

            emb1 = get_emb(os.path.join(images_path, images[0]), transform, net)
            emb2 = get_emb(os.path.join(images_path, images[1]), transform, net)

            distances.append(get_distance(emb1, emb2))

    labels = np.array(labels, dtype=np.bool)
    distances = np.array(distances, dtype=float)
    thresholds = np.arange(0.17, 1.7, 0.001)

    far_list, frr_list, acc_list = [], [], []
    eer = (0, 0)

    for threshold in thresholds:
        far, frr, acc = calculate_metrics(threshold, distances, labels)
        far_list.append(far)
        frr_list.append(frr)
        acc_list.append(acc)
        if round(frr, 3) == round(far, 3):
            eer = (threshold, frr)
            print('Threshold:', round(threshold, 3), 'EER:', round(frr, 3), 'Accuracy:', round(acc, 3))

    draw_plot(thresholds, far_list, frr_list, acc_list, eer)


def get_transform():
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5]),
         ])
    return transform


def load_model(network, weight):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    return net


def get_emb(path, transform, net):
    img = transform(Image.open(path)).unsqueeze(0)
    emb = net(img).detach().numpy()
    return preprocessing.normalize(emb).flatten()


def get_distance(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    dist = np.sqrt(np.sum(np.square(diff)))
    return dist


def calculate_metrics(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    acc = (tp + tn) / dist.size
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    return far, frr, acc


def draw_plot(thresholds, far_list, frr_list, acc_list, eer):
    fig, ax = plt.subplots()
    fig.set_size_inches(14.4, 9.0)

    ax.plot(thresholds, far_list, 'b', label='FAR')
    ax.plot(thresholds, frr_list, 'g', label='FRR')
    ax.plot(thresholds, acc_list, 'r--', label='Accuracy')
    plt.plot(eer[0], eer[1], 'yo', label='EER')
    plt.xlabel('Threshold')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xticks(np.arange(0.15, 1.75, 0.05))
    ax.legend()
    plt.grid(axis='both')

    plt.show()
    fig.savefig('metrics.png', dpi=100)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='datasets/lfw_32', help='Test dataset path.')
    parser.add_argument('-n', '--network', type=str, default='r50', help='backbone network')
    parser.add_argument('-w', '--weight', type=str, default='work_dirs/ms1mv3_r50/ckpt_24.pth')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
