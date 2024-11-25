from filter import generalize_max_median_filter
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error

from noisify import noisify

def evaluate(image, n, noise_type = 'gaussian', **kwargs):
    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    noise_image = None
    if noise_type != 'None':
        noise_image = noisify(image, noise_type, **kwargs)
    filtered_image = []
    for r in range(1,7):
        filtered_image.append(generalize_max_median_filter(noise_image, n, r))

    mean_squared_errors = [mean_squared_error(image, filtered) for filtered in filtered_image]
    plt.plot(range(1, 7), mean_squared_errors)
    plt.xlabel('r')
    plt.ylabel('MSE')
    plt.show()

    return filtered_image, image, noise_image

def plot_images(images, source_image, noise_image=None):
    fig, axs = plt.subplots(1, len(images) + 2, figsize=(20, 10))
    axs[0].imshow(source_image, cmap='gray')
    axs[0].set_title('Original')
    if noise_image is not None:
        axs[1].imshow(noise_image, cmap='gray')
        axs[1].set_title('Noise')
    for i, image in enumerate(images):
        axs[i + 2].imshow(image, cmap='gray')
        axs[i + 2].set_title(f'Filtered r={i + 1}')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--noise_type', type=str, default='None')
    parser.add_argument('--snr', type=float, default=20)
    parser.add_argument('--noise_ratio', type=float, default=0.05)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    images, src_image, noise_img = evaluate(args.image, args.n, args.noise_type, snr=args.snr, noise_ratio=args.noise_ratio)
    plot_images(images, src_image, noise_img)


