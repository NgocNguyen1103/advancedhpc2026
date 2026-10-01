import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

# With CPU
# input_image = plt.imread('/mnt/d/wsl-ubuntu-d/HPC-1/original_056_eaff2b30_frame_000074.jpg')
# width, height, channel = input_image.shape
# pixelCount = width*height
# input_image = input_image.reshape(pixelCount, 3)
# input_image = input_image.copy()
# # for _ in range(len(input_image)):
# #     g = np.uint8((input_image[_][0] + input_image[_][1] + input_image[_][2]) / 3)
# #     input_image[_][0] = input_image[_][1] = input_image[_][2] = g
# # output_image = input_image.reshape(width,height,channel)
# # print(output_image.shape)
# # plt.imshow(output_image)
# # plt.axis('off')
# # plt.savefig('/mnt/d/wsl-ubuntu-d/HPC-1/output_image.png')

#With GPU
input_image = plt.imread('/mnt/d/wsl-ubuntu-d/HPC-1/original_056_eaff2b30_frame_000074.jpg')
width, height, channel = input_image.shape
pixelCount = width*height
input_image = input_image.reshape(pixelCount*channel)
print(input_image.shape)

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

hostDst = np.zeros((width,height, 3), np.uint8)
devSrc = cuda.to_device(input_image)
devDst = cuda.device_array((width, height,3), np.uint8)
blockSize = 64
gridSize = int(pixelCount / blockSize)
grayscale[gridSize,blockSize](devSrc,devDst)
hostDst = devDst.copy_to_host()
