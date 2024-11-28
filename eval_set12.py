from typing import List

from PIL import Image
import numpy as np
from skimage.filters import window

from filter import generalize_max_median_filter
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import root_mean_squared_error
import os
from noisify import noisify

def evaluate_without_noise(image):
    img = Image.open(image)
    img = np.array(img)
    img_holo = hologram(img)
    img_filtered = generalize_max_median_filter(img, 5, 3)
    img_filtered_holo = hologram(img_filtered)

    flot, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original')
    axs[1, 0].imshow(img_filtered, cmap='gray')
    axs[1, 0].set_title('Filtered')
    axs[0, 1].plot(img_holo)
    axs[0, 1].set_title('Hologram')
    axs[1, 1].plot(img_filtered_holo)
    plt.savefig('results/'+image.split('/')[-1].split('.')[0]+'_wo_noise.png')
    plt.close()
    # plt.show()

def hologram(image: np.ndarray):
    img_hologram = [0] * 256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_hologram[round(image[i, j])] += 1

    img_hologram = np.array(img_hologram)
    img_hologram = img_hologram / np.sum(img_hologram)

    return img_hologram

def evaluate_gaussian_noise(image):
    img = Image.open(image)
    img = np.array(img)
    img_holo = hologram(img)
    img_noisy = noisify(img, 'gaussian', snr=10)
    noise_hologram = hologram(img_noisy)
    img_filtered = generalize_max_median_filter(img_noisy, 5, 3)
    filtered_hologram = hologram(img_filtered)

    flot, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(img_noisy, cmap='gray')
    axs[0, 0].set_title('Image with Gaussian Noise')
    axs[1, 0].imshow(img_filtered, cmap='gray')
    axs[1, 0].set_title('Filtered Image')
    axs[0, 1].plot(noise_hologram)
    axs[0, 1].set_title('Hologram')
    axs[1, 1].plot(filtered_hologram)
    axs[0, 1].plot(img_holo)
    axs[0, 1].legend(['Noisy Image', 'Original Image'])
    plt.savefig('results/' + image.split('/')[-1].split('.')[0] + '_gaussian_noise.png')
    plt.close()
    # plt.show()

def evaluate_snp(image):
    img = Image.open(image)
    img = np.array(img)
    img_holo = hologram(img)
    img_noisy = noisify(img, 's&p', noise_ratio=0.1)
    noise_hologram = hologram(img_noisy)
    img_filtered = generalize_max_median_filter(img_noisy, 5, 3)
    filtered_hologram = hologram(img_filtered)

    flot, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(img_noisy, cmap='gray')
    axs[0, 0].set_title('Image with salt & pepper Noise')
    axs[1, 0].imshow(img_filtered, cmap='gray')
    axs[1, 0].set_title('Filtered Image')
    axs[0, 1].set_title('Hologram')

    ln1 = axs[0, 1].twinx().plot(noise_hologram, 'r', label='Noisy Image')
    ln2 = axs[0, 1].plot(img_holo, 'b', label='Original Image')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axs[0, 1].legend(lns, labs, loc=1)
    axs[1, 1].plot(filtered_hologram)

    plt.savefig('results/' + image.split('/')[-1].split('.')[0] + '_s&p.png')
    # plt.show()
    plt.close()

def evaluate_window_size():
    img = Image.open('images/Set12/07.png')
    img = np.array(img)
    noise_img = noisify(img, 's&p', noise_ratio=0.1)
    rMSE = []
    window_size = []
    for i in range(2,11):
        img_filtered = generalize_max_median_filter(noise_img ,i, 3)
        rMSE.append(root_mean_squared_error(img, img_filtered))
        window_size.append(2*i+1)
        if i == 2:
            plt.imshow(img_filtered, cmap='gray')
            plt.savefig('results/window_size_5.png')
            plt.close()
        if i == 10:
            plt.imshow(img_filtered, cmap='gray')
            plt.savefig('results/window_size_21.png')
            plt.close()
    plt.plot(window_size, rMSE)
    plt.xticks(window_size)
    plt.xlabel('Window Size')
    plt.ylabel('rMSE')
    plt.title('Effect of Window Size on the result')
    plt.savefig('results/window_size.png')
    plt.close()

def evaluate_r():
    img = Image.open('images/Set12/05.png')
    img = np.array(img)
    noise_img = noisify(img, 's&p', noise_ratio=0.1)
    rMSE = []
    r = []
    for i in range(1,6):
        img_filtered = generalize_max_median_filter(noise_img ,5, i)
        rMSE.append(root_mean_squared_error(img, img_filtered))
        r.append(i)
        if i == 2:
            plt.imshow(img_filtered, cmap='gray')
            plt.savefig('results/r_2_11.png')
            plt.close()
        if i == 4:
            plt.imshow(img_filtered, cmap='gray')
            plt.savefig('results/r_4_11.png')
            plt.close()

    r = np.array(r)
    rMSE = np.array(rMSE)
    plt.plot(r, rMSE)
    plt.xlabel('r', )
    plt.ylabel('rMSE')
    plt.xticks(r)

    plt.savefig('results/r.png')


# images = os.listdir('images/Set12')
# for image in images:
#     evaluate_without_noise('images/Set12/'+image)
#     evaluate_gaussian_noise('images/Set12/'+image)
#     evaluate_snp('images/Set12/'+image)

evaluate_window_size()
evaluate_r()