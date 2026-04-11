from ._core import Device, FloatTensor as _FloatTensor
CPU = Device.CPU
GPU = Device.GPU

class FloatTensor:
	def __init__(self, _base: _FloatTensor):
		self._base = _base

	# Most functions should pass through to ._core.FloatTensor
	def __getattr__(self, name):
		return getattr(self._base, name)

	@classmethod
	def zeros(cls, shape: list, dev: Device):
		return cls(_FloatTensor.zeros(shape, dev))

	@classmethod
	def from_list(cls, vals: list, dev: Device):
		return cls(_FloatTensor.from_list(vals, dev))

	@classmethod
	def randn(cls, shape, dev: Device):
		return cls(_FloatTensor.randn(shape, dev))

	def __eq__(self, other):
		return self._base.__eq__(other._base)

	def __repr__(self):
		return repr(self._base)

	def __getitem__(self, key):
		if isinstance(key, int):
			key = [key]
		return self._getitem(key)

	def __setitem__(self, key, val):
		if isinstance(key, int):
			key = [key]
		self._setitem(key, val)

	def clone(self):
		return FloatTensor(self._base.clone())

	def view(self, shape):
		return FloatTensor(self._base.view(shape))

	def reshape(self, shape):
		return FloatTensor(self._base.reshape(shape))

	def contiguous(self):
		return FloatTensor(self._base.contiguous())

	def unsqueeze(self, idx):
		return FloatTensor(self._base.unsqueeze(idx))

	def squeeze(self, idx):
		return FloatTensor(self._base.squeeze(idx))

	def permute(self, dims):
		return FloatTensor(self._base.permute(dims))

	def transpose(self, i, j):
		return FloatTensor(self._base.transpose(i, j))

	def flatten(self, start_dim=0, end_dim=self.dim-1):
		return FloatTensor(self._base.flatten(start_dim, end_dim))

	def expand(self, new_shape):
		return FloatTensor(self._base.expand(new_shape))

	def repeat(self, n_repeats):
		return FloatTensor(self._base.repeat(n_repeats))

	def add(self, other):
		return FloatTensor(self._base.add(other._base))

	def sub(self, other):
		return FloatTensor(self._base.sub(other._base))

	def mul(self, other):
		return FloatTensor(self._base.mul(other._base))

	def matmul(self, other):
		return FloatTensor(self._base.matmul(other._base))

	def matmul_3d(self, other):
		return FloatTensor(self._base.matmul_3d(other._base))

	def matmul_tiled(self, other):
		return FloatTensor(self._base.matmul_tiled(other._base))

	def __add__(self, other):
		return self.add(other)

	def __sub__(self, other):
		return self.sub(other)

	def __mul__(self, other):
		return self.mul(other)

	def __matmul__(self, other):
		return self.matmul(other)
