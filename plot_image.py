from matplotlib import pyplot as plt
from math import ceil, sqrt
from torch import Tensor


def plot_image(image: Tensor) -> None:
	num_slices = image.shape[0]
	rows = int(ceil(sqrt(num_slices)))
	cols = int(ceil(num_slices / rows))

	image_width = cols * (image.shape[2] + 2) // 10
	image_height = rows * (image.shape[1] + 2) // 10

	plt.figure("image", (image_width, image_height))
	for row in range(rows):
		for col in range(cols):
			if row * cols + col >= num_slices:
				plt.show()
				return
			plt.subplot(rows, cols, row * cols + col + 1)
			plt.imshow(image[row * cols + col], cmap="gray")
			plt.axis("off")
	plt.show()
