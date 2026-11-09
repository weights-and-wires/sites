class BlockAllocator:
    """Finished for you. Reference-counted physical blocks."""

    def __init__(self, num_blocks, order=None):
        self._free = list(order) if order is not None else list(range(num_blocks))
        self._ref = {}
        self.copies = []          # (source, destination) block copies the block engine would run

    @property
    def num_free(self):
        return len(self._free)

    def allocate(self):
        if not self._free:
            raise MemoryError("no free KV block")
        block = self._free.pop(0)
        self._ref[block] = 1
        return block

    def incref(self, block):
        self._ref[block] += 1

    def ref_count(self, block):
        return self._ref.get(block, 0)

    def free(self, block):
        self._ref[block] -= 1
        if self._ref[block] == 0:
            del self._ref[block]
            self._free.append(block)


class SeqTable:
    def __init__(self, allocator, block_size):
        self.allocator = allocator
        self.block_size = block_size
        self.blocks = []
        self.num_tokens = 0

    def fork(self):
        """New table that shares every block with this one; no data is copied."""
        raise NotImplementedError

    def append_token(self):
        """Store one token; return (physical_block, offset). Copy the last block first if it is shared."""
        raise NotImplementedError

    def release(self):
        """Finished for you."""
        for block in self.blocks:
            self.allocator.free(block)
        self.blocks = []
        self.num_tokens = 0
