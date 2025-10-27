class Interval:
    def __init__(self, center, length):
        self.center = center
        self.length = length
        self.start = center - length
        self.end = center + length
    def __repr__(self):
        return f"[{self.start}, {self.end}]"

class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []
        self.next = None
        self.prev = None

class SNBTree:
    def __init__(self, degree=4):
        self.root = Node(is_leaf=True)
        self.degree = degree

    def insert(self, center, length):
        interval = Interval(center, length)
        res = self._insert_recursive(self.root, interval)
        if res:
            key, new_node = res
            new_root = Node(is_leaf=False)
            new_root.keys = [key]
            new_root.children = [self.root, new_node]
            self.root = new_root

    def _insert_recursive(self, node, interval):
        if node.is_leaf:
            return self._insert_into_leaf(node, interval)
        i = 0
        while i < len(node.keys) and interval.start >= node.keys[i]:
            i += 1
        res = self._insert_recursive(node.children[i], interval)
        if not res:
            return None
        key, new_child = res
        j = 0
        while j < len(node.keys) and key >= node.keys[j]:
            j += 1
        node.keys.insert(j, key)
        node.children.insert(j+1, new_child)
        if len(node.children) > self.degree:
            return self._split_internal(node)
        return None

    def _insert_into_leaf(self, leaf, interval):
        i = 0
        while i < len(leaf.keys) and leaf.keys[i].start < interval.start:
            i += 1
        leaf.keys.insert(i, interval)
        if len(leaf.keys) < self.degree:
            return None
        mid = (len(leaf.keys) + 1) // 2
        new_leaf = Node(is_leaf=True)
        new_leaf.keys = leaf.keys[mid:]
        leaf.keys = leaf.keys[:mid]
        new_leaf.next = leaf.next
        if new_leaf.next:
            new_leaf.next.prev = new_leaf
        leaf.next = new_leaf
        new_leaf.prev = leaf
        return (new_leaf.keys[0].start, new_leaf)

    def _split_internal(self, node):
        mid = len(node.keys) // 2
        key_to_parent = node.keys[mid]
        new_node = Node(is_leaf=False)
        new_node.keys = node.keys[mid+1:]
        new_node.children = node.children[mid+1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid+1]
        return (key_to_parent, new_node)

    def find(self, key):
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        lo, hi = 0, len(node.keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if node.keys[mid].start <= key:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        for j in range(idx-1, -1, -1):
            interval = node.keys[j]
            if interval.start <= key <= interval.end:
                return interval
        cur = node.prev
        while cur:
            for j in range(len(cur.keys)-1, -1, -1):
                interval = cur.keys[j]
                if interval.start <= key <= interval.end:
                    return interval
            cur = cur.prev
        return None

    def range_query(self, start, end):
        results = []
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and start >= node.keys[i]:
                i += 1
            node = node.children[i]
        lo, hi = 0, len(node.keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if node.keys[mid].start < start:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        back_list = []
        cur_leaf = node
        j = idx - 1
        while True:
            if j < 0:
                if cur_leaf.prev:
                    cur_leaf = cur_leaf.prev
                    j = len(cur_leaf.keys) - 1
                    continue
                break
            interval = cur_leaf.keys[j]
            if interval.end >= start:
                back_list.append(interval)
            j -= 1
        back_list.reverse()
        results.extend(back_list)
        cur_leaf = node
        i = idx
        while True:
            if i < len(cur_leaf.keys):
                interval = cur_leaf.keys[i]
                if interval.start > end:
                    break
                if interval.end >= start:
                    results.append(interval)
                i += 1
                continue
            if cur_leaf.next is None:
                break
            cur_leaf = cur_leaf.next
            i = 0
            if cur_leaf.keys[0].start > end:
                break
        return results
