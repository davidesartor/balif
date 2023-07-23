from dataclasses import dataclass
from functools import cached_property
import math
from typing import Self # type: ignore

def depth(node_idx) -> int:
    return math.frexp(node_idx + 1)[1] - 1

def left_child_idx(parent_idx) -> int:
    return 2 * parent_idx + 1

def right_child_idx(parent_idx) -> int:
    return 2 * parent_idx + 2

def path_to(node_idx) -> list[int]:
    """Indices of nodes on the path from the root to the current node (both included)."""
    path = [node_idx]
    while node_idx > 0:
        node_idx = (node_idx - 1) // 2
        path.append(node_idx)
    return path[::-1]


@dataclass(frozen=True, kw_only=True)
class BinaryTreeNode:
    idx: int
    children: tuple[Self, Self] | None = None

    @cached_property
    def depth(self) -> int:
        return math.frexp(self.idx + 1)[1] - 1

    @cached_property
    def path_to(self) -> list[int]:
        """Indices of nodes on the path from the root to the current node (both included)."""
        return path_to(self.idx)
    
    @property
    def is_leaf(self) -> bool:
        return self.children is None
    
    
    