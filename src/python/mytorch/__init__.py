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
		if isinstance(shape, int):
			shape = (shape,)
		return cls(_FloatTensor.zeros(shape, dev))

	@classmethod
	def from_list(cls, vals: list, dev: Device):
		return cls(_FloatTensor.from_list(vals, dev))

	@classmethod
	def randn(cls, shape, dev: Device):
		if isinstance(shape, int):
			shape = (shape,)
		return cls(_FloatTensor.randn(shape, dev))

	def __eq__(self, other):
		return self._base.__eq__(other._base)

	def __repr__(self):
		return repr(self._base)

	# Supports both single-entry indexing (e.g. x[3, 2])
	# and range-based indexing (e.g. x[:, 2:3]).
	# Range-based indexing returns a view of self.
	def __getitem__(self, key):
		if isinstance(key, int) or isinstance(key, slice):
			key = (key,)
		if all(isinstance(item, int) for item in key) and len(key) == self.dim:
			return self._getitem(key)
		# key should be a list of ints and slice objects
		assert isinstance(key, tuple), "Key must be a tuple of ints and slice objects"

		if len(key) > self.dim:
			raise ValueError("Index into tensor cannot be longer than dimension of tensor.")
		# build inputs to indexed_view
		singleton = []
		shape = []
		rel_offsets = []
		rel_strides = []

		for idx, item in enumerate(key):
			if isinstance(item, int):
				singleton.append(True)
				shape.append(1)
				rel_offsets.append(item)
				rel_strides.append(0)
			elif isinstance(item, slice):
				singleton.append(False)

				# compute shape, offset, stride
				item_indices = item.indices(self.shape[idx])
				this_shape = len(range(*item_indices))
				this_offset = item_indices[0]
				this_stride = item_indices[2]

				shape.append(this_shape)
				rel_offsets.append(this_offset)
				rel_strides.append(this_stride)

			else:
				raise NotImplementedError("Key must be a list of ints and slice objects.")

		assert len(singleton) == len(shape) == len(rel_offsets) == len(rel_strides)

		# if index only includes some dims, view the full dim for remaining dims
		while len(singleton) < self.dim:
			idx += 1     # idx is still in scope
			singleton.append(False)
			shape.append(self.shape[idx])
			rel_offsets.append(0)
			rel_strides.append(1)

		return FloatTensor(self._base.indexed_view(singleton, shape, rel_offsets, rel_strides))
				


	def __setitem__(self, key, val):
		if isinstance(key, int) or isinstance(key, slice):
			key = (key,)
		if all(isinstance(item, int) for item in key) and len(key) == self.dim:
			self._setitem(key, val)
			return
		# use __getitem__ to get the view
		temp_view = self.__getitem__(key)
		# then make the assignment
		temp_view.view_assign(val)

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

	def flatten(self, start_dim=None, end_dim=None):
		if start_dim is None:
			start_dim = 0
		if end_dim is None:
			end_dim = self.dim - 1
		return FloatTensor(self._base.flatten(start_dim, end_dim))

	def expand(self, new_shape):
		return FloatTensor(self._base.expand(new_shape))

	def repeat(self, n_repeats):
		return FloatTensor(self._base.repeat(n_repeats))

	@classmethod
	def cat(cls, tensors: list[FloatTensor], dim=0):
		assert len(tensors) > 0, "cat() needs at least one tensor to concatenate!"

		new_len = 0

		if dim < 0:
			dim += tensors[0].dim
		if dim < 0 or dim >= tensors[0].dim:
			raise ValueError("Dimension out of range.")

		for tensor in tensors:
			assert tensor.dim == tensors[0].dim, "All tensors in cat() must have same dimension"
			for idx in range(tensors[0].dim):
				assert idx == dim or tensor.shape[idx] == tensors[0].shape[idx], (
						"All tensors in cat() must have same shape along non-concatenated dimensions"
						)
			new_len += tensor.shape[dim]


		new_shape = tensors[0].shape.copy()
		new_shape[dim] = new_len

		result = FloatTensor.zeros(new_shape, tensors[0].device)

		# "Row" is not the best word, "row_idx" means the index along the concatenated dimension
		row_idx = 0
		view_idx_slices = [slice(None, None, None) for _ in range(tensors[0].dim)]
		for tensor in tensors:
			# find the tuple of slices to describe the view where tensor will get inserted
			n_rows_this_tensor = tensor.shape[dim]
			rows_slice = slice(row_idx, row_idx + n_rows_this_tensor, None)
			view_idx_slices[dim] = rows_slice
			
			# great, now we can make the assignment
			result[tuple(view_idx_slices)] = tensor

			row_idx += n_rows_this_tensor

		# woohoo, all done
		return result

	@classmethod
	def stack(cls, tensors: list[FloatTensor], dim=0):
		# Hack: stack() is just unsqueeze() all the tensors, followed by cat().
		unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors]
		return FloatTensor.cat(unsqueezed_tensors, dim)

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

	def abs(self):
		return FloatTensor(self._base.abs())

	def exp(self):
		return FloatTensor(self._base.exp())

	def log(self):
		return FloatTensor(self._base.log())

	def sqrt(self):
		return FloatTensor(self._base.sqrt())

	def relu(self):
		return FloatTensor(self._base.relu())

	def sum(self, dim):
		return FloatTensor(self._base.sum(dim))

