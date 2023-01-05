from matplotlib import pyplot as plt
from math import ceil, sqrt
import numpy as np
from torch import Tensor
import plotly.graph_objects as go


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


def plot_volume(attention_map):
	X, Y, Z = np.mgrid[0:28, 0:28, 0:28]
	values = attention_map

	fig = go.Figure(data=go.Volume(
		x=X.flatten(),
		y=Y.flatten(),
		z=Z.flatten(),
		value=values.flatten(),
		isomin=-0.1,
		isomax=0.8,
		opacity=0.1,  # needs to be small to see through all surfaces
		surface_count=21,  # needs to be a large number for good volume rendering
		colorscale='RdBu'
	))
	fig.show()
