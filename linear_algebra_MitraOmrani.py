# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import random
import matplotlib.pyplot as plt
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

image_dir = '/kaggle/input/image-classification/images/images/architecure'

# width = 1000
# height = 1000

num_image = 5
image_file = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

selecetd_images = random.sample(image_file,num_image)

def box_muller_transform(mu, sigma):
    u1 = random.random()
    u2 = random.random()

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)

    return z0 * sigma + mu, z1 * sigma + mu

def add_gaussian_noise(image, mu=0, sigma=1):
    r = image[:, :, 0]
    g = image[:, :, 1] 
    b = image[:, :, 2]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r[i][j] += box_muller_transform(mu, sigma)[0]
            
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            g[i][j] += box_muller_transform(mu, sigma)[0]
            
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b[i][j] += box_muller_transform(mu, sigma)[0]
    
    # Clip the values to the valid range
    image = np.clip(image, 0, 255)
    return image

def gram_schmidt(A):
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((n, n))

    for k in range(n):
        R[k, k] = np.linalg.norm(A[:, k])
        if R[k, k] > 1e-6:  # Check if the norm is close to zero
            Q[:, k] = A[:, k] / R[k, k]
        else:
            Q[:, k] = np.zeros_like(Q[:, k])  # If close to zero, set the vector to zero

        for j in range(k+1, n):
            R[k, j] = np.dot(Q[:, k].T, A[:, j])
            A[:, j] = A[:, j] - R[k, j] * Q[:, k]

    return Q, R

def power_method(A, max_iter=50):
    n = A.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        Av = np.dot(A, v)
        v_new = Av / np.linalg.norm(Av)
        if np.abs(np.dot(v, v_new) - 1) < 1e-6:
            break
        v = v_new
    return v_new

def qr_eig(A, max_iter=1000):
    for _ in range(max_iter):
        Q, R = gram_schmidt(A)
        A = np.dot(R, Q)
    eigenvalues = np.diag(A)
    eigenvectors = []
    for eig in eigenvalues:
        A_eig = A - eig * np.eye(A.shape[0])
        eigenvectors.append(power_method(A_eig))
    return eigenvalues, np.array(eigenvectors).T

def calculate_SVD1(A):
    # Step 1: Compute the covariance matrix
    C = np.dot(A.T, A)

    # Step 2: Perform eigenvalue decomposition
#     eigenvalues, eigenvectors = eigen_decomposition(C)
    eigenvalues, eigenvectors =qr_eig(C,100)
    # eigenvalues, eigenvectors = power_iteration(C)
    eigenvalues, eigenvectors =np.linalg.eig(C)
#     eigenvalues, eigenvectors = eigen_decomposition(C)
    

    # Step 3: Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 4: Calculate singular values
    eigenvalues = np.abs(eigenvalues)
    singular_values = np.sqrt(eigenvalues)

    # Step 5: Calculate left singular vectors
    U = np.dot(A, eigenvectors) / singular_values

    # Step 6: Normalize right singular vectors
    VT = eigenvectors.T

    return U, singular_values, VT

def norm(vector):
    norm = 0.0

    for i in range(len(vector)):
        norm += vector[i] * vector[i]

    norm = math.sqrt(norm)

    return norm

def gram_schmidt(A):
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((n, n))

    for k in range(n):
        R[k, k] = np.linalg.norm(A[:, k])
        if R[k, k] > 1e-6:  # Check if the norm is close to zero
            Q[:, k] = A[:, k] / R[k, k]
        else:
            Q[:, k] = np.zeros_like(Q[:, k])  # If close to zero, set the vector to zero

        for j in range(k+1, n):
            R[k, j] = np.dot(Q[:, k].T, A[:, j])
            A[:, j] = A[:, j] - R[k, j] * Q[:, k]

    return Q, R

def denoise_channel(channel, threshold):
#     channel = np.matrix(channel)
    U, s, Vt = calculate_SVD1(channel)

#     np.linalg.svd(channel, full_matrices=False)
#     U, s, Vt = np.linalg.svd(channel, full_matrices=False)

    # Thresholding
    print(s)
    s[s < threshold] = 0

    # Reconstruct the channel

    denoised_channel = U.dot(np.diag(s)).dot(Vt)

    return denoised_channel

def denoise(image,threshold):
    b, g, r = cv2.split(image)
    # Set the threshold for denoising

    # Perform denoising on each color channel
    denoised_b = denoise_channel(b, threshold)
    denoised_g = denoise_channel(g, threshold)
    denoised_r = denoise_channel(r, threshold)
    
    # Merge the denoised color channels back into an RGB image
    denoised_image = cv2.merge((denoised_b, denoised_g, denoised_r))

    # Clip the pixel values to ensure they are within the valid range
#     denoised_image = np.clip(denoised_image, 0, 255)

#     Convert the pixel values back to integers
    denoised_image = denoised_image.astype(np.uint8)

    return denoised_image

width = 800
height = 800
from PIL import Image
for filename in selecetd_images:
# #     print(filename)
#     image = Image.open('/kaggle/input/image-classification/images/images/architecure/2447124918_d795e9d37d_n.jpg')
#     Image.fromarray(add_gaussian_noise(np.array(image), 0, 50)).show()
#     Image.fromarray(add_gaussian_noise(np.array(image), 0, 10)).show()
    
    img = cv2.imread(os.path.join(image_dir, filename))
    img = cv2.resize(img,(width, height))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    
#     # Add Gaussian noise
    noisy_image = add_gaussian_noise(img, 0,50)
#     Image.fromarray(add_gaussian_noise(np.array(img), 0, 50)).show()
    plt.imshow(noisy_image)
    plt.show()

    denoised_image = denoise(noisy_image,40)
#     print(i)
    plt.imshow(denoised_image)
    plt.show()

