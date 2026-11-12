class Node:
    """Finished for you. One radix-tree node that owns the tokens on the edge above it."""

    def __init__(self, edge=(), parent=None):
        self.edge = tuple(edge)
        self.children = {}        # first token of the child's edge -> child
        self.parent = parent
        self.ref = 0              # number of running requests that use this node
        self.last_access = 0


class RadixCache:
    """Finished for you. A radix tree of cached tokens with a token budget (Section 3)."""

    def __init__(self, capacity):
        self.root = Node()
        self.capacity = capacity
        self.clock = 0

    def _nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def size(self):
        return sum(len(n.edge) for n in self._nodes())

    def peek(self, tokens):
        """Length of the longest cached prefix of tokens. Changes nothing."""
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            k = 0
            while k < len(child.edge) and i + k < len(tokens) and child.edge[k] == tokens[i + k]:
                k += 1
            i += k
            if k < len(child.edge):
                break
            node = child
        return i

    def insert(self, tokens):
        """Store tokens, splitting an edge where they diverge. Touches every node on the path."""
        tokens = tuple(tokens)
        self.clock += 1
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                leaf = Node(tokens[i:], node)
                leaf.last_access = self.clock
                node.children[tokens[i]] = leaf
                return
            k = 0
            while k < len(child.edge) and i + k < len(tokens) and child.edge[k] == tokens[i + k]:
                k += 1
            if k < len(child.edge):
                mid = Node(child.edge[:k], node)
                mid.ref = child.ref                  # every user of the child also uses mid
                mid.last_access = child.last_access
                node.children[tokens[i]] = mid
                child.edge = child.edge[k:]
                child.parent = mid
                mid.children[child.edge[0]] = child
                child = mid
            child.last_access = self.clock
            i += k
            node = child

    def _path(self, tokens):
        node, i, path = self.root, 0, []
        while i < len(tokens):
            node = node.children[tokens[i]]
            path.append(node)
            i += len(node.edge)
        return path

    def lock(self, tokens):
        for node in self._path(tokens):
            node.ref += 1

    def unlock(self, tokens):
        for node in self._path(tokens):
            node.ref -= 1

    def evict(self, num_tokens):
        """Free at least num_tokens by removing least recently used leaves with ref == 0."""
        freed = 0
        while freed < num_tokens:
            leaves = [n for n in self._nodes() if n is not self.root and not n.children and n.ref == 0]
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            del victim.parent.children[victim.edge[0]]
            freed += len(victim.edge)
        return freed

    def admit(self, tokens):
        """Finished for you. Serve one request: return the cached prefix length, then store it.

        The request's path is locked while the cache evicts down to the budget,
        so it never evicts the tokens it is about to use.
        """
        tokens = tuple(tokens)
        matched = self.peek(tokens)
        self.insert(tokens)
        self.lock(tokens)
        over = self.size() - self.capacity
        if over > 0:
            self.evict(over)
        self.unlock(tokens)
        return matched


def pick_next(waiting, cache, policy):
    """Return the position in `waiting` of the request to serve next.

    `waiting` is a list of token tuples in arrival order.
    policy "fcfs": the oldest request, position 0.
    policy "lpm": the request with the longest cached prefix, `cache.peek(request)`.
    On a tie, the request that arrived first wins.
    """
    raise NotImplementedError


def serve(requests, capacity, policy):
    """Serve every request one at a time and return (hit_tokens, total_tokens, order).

    Start from an empty RadixCache(capacity). Keep a list of waiting requests.
    While it is not empty, choose one with pick_next, remove it, serve it with
    cache.admit, add the returned prefix length to hit_tokens, and add the
    request's length to total_tokens. `order` lists the original indices of the
    requests in the order they were served.
    """
    raise NotImplementedError
