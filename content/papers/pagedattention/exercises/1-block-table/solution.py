class BlockAllocator:
    """Physical KV blocks with reference counts (Section 4.2 and Section 4.4)."""

    def __init__(self, num_blocks, order=None):
        self._free = list(order) if order is not None else list(range(num_blocks))
        self._ref = {}

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
        if block not in self._ref:
            raise ValueError(f"block {block} is not allocated")
        self._ref[block] += 1

    def ref_count(self, block):
        return self._ref.get(block, 0)

    def free(self, block):
        if block not in self._ref:
            raise ValueError(f"block {block} is not allocated")
        self._ref[block] -= 1
        if self._ref[block] == 0:
            del self._ref[block]
            self._free.append(block)


class BlockTable:
    """The block table of one sequence: logical block i -> physical block blocks[i]."""

    def __init__(self, allocator, block_size):
        self.allocator = allocator
        self.block_size = block_size
        self.blocks = []
        self.num_tokens = 0

    @property
    def wasted_slots(self):
        return len(self.blocks) * self.block_size - self.num_tokens

    def append_token(self):
        if self.num_tokens % self.block_size == 0:
            self.blocks.append(self.allocator.allocate())
        offset = self.num_tokens % self.block_size
        self.num_tokens += 1
        return self.blocks[-1], offset

    def translate(self, pos):
        if not 0 <= pos < self.num_tokens:
            raise IndexError(f"position {pos} is outside 0..{self.num_tokens - 1}")
        return self.blocks[pos // self.block_size], pos % self.block_size

    def release(self):
        for block in self.blocks:
            self.allocator.free(block)
        self.blocks = []
        self.num_tokens = 0
