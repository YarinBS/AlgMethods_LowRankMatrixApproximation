from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def K_estimation(mat, k, n, m):
    U, s, Vt = np.linalg.svd(mat)
    Sigma = np.zeros((m, n))
    for i in range(s.size):
        Sigma[i, i] = s[i]

    approx = U @ Sigma[:, :k] @ Vt[:k, :]
    return approx


img = Image.open('pic1.jpg')

print(f"Orignal image format: {img.format}, Size: {img.size}, Mode: {img.mode}\n")
k_values = [200, 250, 300, 350, 400, 500, 550, 600, 650, 700]
n, m = img.size[0], img.size[1]
pix = np.array(img)

red_array = pix[:, :, 0]
green_array = pix[:, :, 1]
blue_array = pix[:, :, 2]

for k in k_values:
    K_red_ARR = K_estimation(red_array, k, n, m)
    K_green_ARR = K_estimation(green_array, k, n, m)
    K_blue_ARR = K_estimation(blue_array, k, n, m)

    constructedMat = np.dstack((K_red_ARR, K_green_ARR, K_blue_ARR))
    image_k = Image.fromarray(np.uint8(constructedMat))
    plt.imshow(image_k, cmap="gray")
    numerator = np.linalg.norm(red_array - K_red_ARR, "fro") ** 2
    denominator = np.linalg.norm(red_array, "fro") ** 2
    print(f"Relative approximation error for k={k} is {numerator / denominator}. Picture:")
    plt.show()
    print()
