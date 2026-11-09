class BlockAllocator:
    """Physical KV blocks with reference counts (Section 4.2 and Section 4.4)."""

    def __init__(self, num_blocks, order=None):
        raise NotImplementedError

    @property
    def num_free(self):
        """Number of blocks on the free list."""
        raise NotImplementedError

    def allocate(self):
        """Pop the front of the free list, set its count to 1, and return it. MemoryError if empty."""
        raise NotImplementedError

    def incref(self, block):
        """Add one reference to an allocated block. ValueError if it is free."""
        raise NotImplementedError

    def ref_count(self, block):
        """Reference count of a block; 0 when it is free."""
        raise NotImplementedError

    def free(self, block):
        """Drop one reference; at 0 the block goes to the back of the free list. ValueError if it is free."""
        raise NotImplementedError


class BlockTable:
    """The block table of one sequence: logical block i -> physical block blocks[i]."""

    def __init__(self, allocator, block_size):
        self.allocator = allocator
        self.block_size = block_size
        self.blocks = []
        self.num_tokens = 0

    @property
    def wasted_slots(self):
        """Allocated slots that hold no token yet."""
        raise NotImplementedError

    def append_token(self):
        """Store one more token; return (physical_block, offset). Allocate only when the last block is full."""
        raise NotImplementedError

    def translate(self, pos):
        """Return (physical_block, offset) of the token at position pos. IndexError if out of range."""
        raise NotImplementedError

    def release(self):
        """Free every block and reset the table."""
        raise NotImplementedError
