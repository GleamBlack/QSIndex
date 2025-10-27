class SNBTree:
    class Node:
        def __init__(self, leaf=False):
            self.leaf = leaf
            self.keys = []       
            self.children = []    
            self.intervals = []    
            self.parent = None   

    class Interval:
        def __init__(self, center, length):
            """根据中心和长度初始化区间，计算起始和结束位置"""
            self.center = center
            self.length = length
            self.start = center - length / 2.0
            self.end = center + length / 2.0

        def __repr__(self):
            return f"[{self.start}, {self.end}]"

    def __init__(self, order=4):
        self.order = order
        self.max_children = order               
        self.max_leaf_items = order - 1         
        self.root = self.Node(leaf=True)

    def _find_leaf(self, node, key):
        if node.leaf:
            return node
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        return self._find_leaf(node.children[i], key)

    def insert(self, center, length):
        interval = self.Interval(center, length)
        leaf = self._find_leaf(self.root, interval.start)
        i = 0
        while i < len(leaf.intervals) and leaf.intervals[i].start < interval.start:
            i += 1
        leaf.intervals.insert(i, interval)
        if len(leaf.intervals) > self.max_leaf_items:
            self._split_leaf(leaf)

    def _split_leaf(self, leaf):
        new_leaf = self.Node(leaf=True)
        new_leaf.next = leaf.next
        leaf.next = new_leaf
        total = len(leaf.intervals)
        left_count = (total + 1) // 2         
        right_count = total - left_count       
        new_leaf.intervals = leaf.intervals[left_count:]   
        leaf.intervals = leaf.intervals[:left_count]       
        new_key = new_leaf.intervals[0].start 
        new_leaf.parent = leaf.parent
        if leaf.parent is None:
            new_root = self.Node(leaf=False)
            new_root.keys = [new_key]
            new_root.children = [leaf, new_leaf]
            leaf.parent = new_root
            new_leaf.parent = new_root
            self.root = new_root
        else:
            parent = leaf.parent
            idx = parent.children.index(leaf)
            parent.children.insert(idx + 1, new_leaf)
            parent.keys.insert(idx, new_key)
            new_leaf.parent = parent
            if len(parent.children) > self.max_children:
                self._split_internal(parent)

    def _split_internal(self, node):
        new_node = self.Node(leaf=False)
        total_children = len(node.children)
        left_count = (total_children + 1) // 2    
        right_count = total_children - left_count
        mid_key_index = left_count - 1
        up_key = node.keys[mid_key_index]        
        left_keys = node.keys[:mid_key_index]
        right_keys = node.keys[mid_key_index + 1:]
        left_children = node.children[:left_count]
        right_children = node.children[left_count:]
        node.keys = left_keys
        node.children = left_children
        for child in node.children:
            child.parent = node
        new_node.keys = right_keys
        new_node.children = right_children
        for child in new_node.children:
            child.parent = new_node
        new_node.parent = node.parent
        if node.parent is None:
            new_root = self.Node(leaf=False)
            new_root.keys = [up_key]
            new_root.children = [node, new_node]
            node.parent = new_root
            new_node.parent = new_root
            self.root = new_root
        else:
            parent = node.parent
            idx = parent.children.index(node)
            parent.children.insert(idx + 1, new_node)
            parent.keys.insert(idx, up_key)
            new_node.parent = parent
            if len(parent.children) > self.max_children:
                self._split_internal(parent)

    def find(self, key):
        leaf = self._find_leaf(self.root, key)
        for interval in leaf.intervals:
            if interval.start <= key <= interval.end:
                return interval
        return None

    def range_query(self, query_start, query_end):
        result = []
        leaf = self._find_leaf(self.root, query_start)
        # 从该叶节点开始顺序扫描
        while leaf is not None:
            for interval in leaf.intervals:
                if interval.start > query_end:
                    return result
                if interval.end < query_start:
                    continue
                result.append(interval)
            leaf = leaf.next
        return result
