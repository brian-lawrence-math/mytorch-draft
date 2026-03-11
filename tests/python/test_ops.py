from mytorch import FloatTensor as FT
from mytorch import Device, CPU, GPU

import pytest

def approx_eq(x, y, eps = 1e-4):
	return abs(y - x) < eps

def test_add_on_cpu():
	x_list = [1, 2, 3, 4]
	y_list = [0.4, 0.3, 0.2, 0.1]
	x = FT.from_list(x_list, CPU)
	y = FT.from_list(y_list, CPU)
	z = x.add(y)
	for i in range(4):
		assert approx_eq(z[[i]], x_list[i] + y_list[i])

def test_add_on_gpu():
	x_list = [1, 2, 3, 4]
	y_list = [0.4, 0.3, 0.2, 0.1]
	x = FT.from_list(x_list, GPU)
	y = FT.from_list(y_list, GPU)
	z = x.add(y)
	for i in range(4):
		assert approx_eq(z[[i]], x_list[i] + y_list[i])

def test_add_with_moves():
	x_list = [1, 2, 3, 4]
	y_list = [0.4, 0.3, 0.2, 0.1]
	x = FT.from_list(x_list, CPU)
	y = FT.from_list(y_list, GPU)

	# put both on GPU and do the addition there
	x.move_to_device(GPU)
	z = x.add(y)
	
	# back to CPU
	z.move_to_device(CPU)
	for i in range(4):
		assert approx_eq(z[[i]], x_list[i] + y_list[i])


