class Node:
    """Finished for you. One radix-tree node that owns the tokens on the edge above it."""

    def __init__(self, edge=(), parent=None):
        self.edge = tuple(edge)
        self.children = {}        # first token of the child's edge -> child
        self.parent = parent


class RadixTree:
    """A radix tree over token ids (Section 3). Edges carry sequences of tokens."""

    def __init__(self):
        self.root = Node()

    def _nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def num_nodes(self):
        """Finished for you. Number of nodes, not counting the root."""
        return sum(1 for _ in self._nodes()) - 1

    def num_tokens(self):
        """Finished for you. Number of tokens stored on all edges."""
        return sum(len(n.edge) for n in self._nodes())

    def match(self, tokens):
        """Return the length of the longest prefix of `tokens` stored in the tree.

        The prefix can end in the middle of an edge. This method changes nothing.
        """
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
        """Store `tokens` and return how many tokens were new (not already cached).

        Split an edge where `tokens` diverges from it, or ends inside it.
        """
        tokens = tuple(tokens)
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                node.children[tokens[i]] = Node(tokens[i:], node)
                return len(tokens) - i
            k = 0
            while k < len(child.edge) and i + k < len(tokens) and child.edge[k] == tokens[i + k]:
                k += 1
            if k < len(child.edge):
                mid = Node(child.edge[:k], node)
                node.children[tokens[i]] = mid
                child.edge = child.edge[k:]
                child.parent = mid
                mid.children[child.edge[0]] = child
                child = mid
            i += k
            node = child
        return 0
