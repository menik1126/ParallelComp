import queue
import random
import itertools
class FixedSizePriorityQueue:
    def __init__(self, size=5, key="priority"):
        self.size = size
        self.key = key  # 用来决定优先级的字典键
        self.pq = queue.PriorityQueue()
        self.counter = itertools.count()

    def add(self, element):
        # 将字典中的某个key的值作为优先级
        priority_value = element.get(self.key, float('inf'))  # 默认值是inf，表示最小的优先级
        self.pq.put((priority_value, next(self.counter), element))  # 插入元素，优先级和字典作为元组

        # 如果队列大小超过设定的最大值，移除优先级最低的元素
        if self.pq.qsize() > self.size:
            self.pq.get()

    def get_elements_ascending(self):
        return sorted([item[2] for item in self.pq.queue], key=lambda x: x[self.key])

    def get_elements_ascending_ids(self):
        sorted_list = sorted(self.pq.queue, key=lambda item: item[0]) 
        
        return [item[2]["raw_location"] for item in sorted_list]
    
    def get_elements_random(self):
        items = [item[2] for item in self.pq.queue]  # 提取需要的元素
        random.shuffle(items)
        return items
    
    def get_elements_descending(self):
        return sorted([item[2] for item in self.pq.queue], key=lambda x: x[self.key], reverse=True)

    def get_elements(self):
        return [item[2] for item in self.pq.queue]

    def get_elements_key(self,key):
        return sorted([item[2] for item in self.pq.queue], key=lambda x: x[key])
    
# 示例
# pq = FixedSizePriorityQueue(size=5, key="priority")

# # 添加字典元素
# elements = [
#     {"name": "A", "priority": 10, "haha": 1},
#     {"name": "B", "priority": 20, "haha": 1},
#     {"name": "C", "priority": 15, "haha": 1},
#     {"name": "D", "priority": 30, "haha": 1},
#     {"name": "E", "priority": 25, "haha": 1},
#     {"name": "F", "priority": 5, "haha": 1},
#     {"name": "G", "priority": 35, "haha": 1}
# ]

# for elem in elements:
#     pq.add(elem)
#     str_key = 'name'
#     print(f"Queue after adding {elem}: {pq.get_elements_key(str_key)} ")
