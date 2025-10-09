import torch

class TensorBuffer:
    """
    Class to flatten and deflatten the gradient vector.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._len_tensors = len(tensors)
        self._tensor_shapes = [tensor.size() for tensor in tensors]

        self.buffer = torch.cat([tensor.view(-1) for tensor in tensors])

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(self._tensor_shapes[index])

    def __len__(self):
        return self._len_tensors
    
    def deflatten(self):
        """将 buffer 拆分成原始形状的张量列表"""
        return [
            self.buffer[start:end].view(shape)
            for start, end, shape in zip(
                self._start_idx, self._end_idx, self._tensor_shapes
            )
        ]
def chunk(tensor: torch.Tensor, chunk_size: int):
    original_shape = tensor.shape
    chunks = torch.split(tensor, chunk_size, dim=0)
    return chunks, original_shape

def combine(chunks, original_shape) -> torch.Tensor:
    combined = torch.cat(chunks, dim=0)
    assert combined.shape == original_shape, (
        f"shape{combined.shape} after combined differ with the previous shape{original_shape}"
    )
    return combined
