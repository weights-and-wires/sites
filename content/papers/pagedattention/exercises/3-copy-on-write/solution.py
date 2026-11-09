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
        child = SeqTable(self.allocator, self.block_size)
        child.blocks = list(self.blocks)
        child.num_tokens = self.num_tokens
        for block in self.blocks:
            self.allocator.incref(block)
        return child

    def append_token(self):
        """Store one token; return (physical_block, offset). Copy the last block first if it is shared."""
        offset = self.num_tokens % self.block_size
        if offset == 0:
            self.blocks.append(self.allocator.allocate())
        elif self.allocator.ref_count(self.blocks[-1]) > 1:
            old = self.blocks[-1]
            new = self.allocator.allocate()
            self.allocator.copies.append((old, new))
            self.allocator.free(old)
            self.blocks[-1] = new
        self.num_tokens += 1
        return self.blocks[-1], offset

    def release(self):
        """Finished for you."""
        for block in self.blocks:
            self.allocator.free(block)
        self.blocks = []
        self.num_tokens = 0
