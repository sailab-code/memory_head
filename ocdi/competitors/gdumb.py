from competitors import CompetitorWrapperER
import random
from torch import nn
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


#
#
# class GDumb(CompetitorWrapperER):
#     def __init__(self, net, buffer_size, input_size):
#         super(GDumb, self).__init__(net, buffer_size, input_size)
#         self.class_memory = {}  # memory dict = {0: [ex0, ex1], 1: [ex0, ex1], 2: []}
#         self.memory_class_counter = {}  # class dict = {0: 10, 1: 2}  - counter for each class
#
#     def update_buffer(self, x, y):
#         k_c = self.buffer_size // max(1, len(self.memory_class_counter))
#         mask = (self.memory_class_counter[y] < k_c) | (self.class_memory[y] == [])
#         replace_mask = self.memory_class_counter.sum() >= self.buffer_size
#         replace_idx = torch.argmax(self.memory_class_counter) if replace_mask else None
#         if replace_mask:
#             replace_idx = torch.randint(0, self.memory_class_counter[replace_idx], (1,)).item()
#             self.class_memory[replace_idx].pop()
#             self.memory_class_counter[replace_idx] -= 1
#         if mask:
#             self.class_memory[y].append(x[mask].cpu().numpy())
#             self.memory_class_counter[y] += mask.sum()


# define dataset class
class MemoryDataset(Dataset):
    def __init__(self, mem_x, mem_y):
        self.mem_x = mem_x
        self.mem_y = mem_y

    def __len__(self):
        return len(self.mem_x)

    def __getitem__(self, idx):
        return self.mem_x[idx], self.mem_y[idx]


# function to reset parameters
def _reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class GDumb(CompetitorWrapperER):
    def __init__(self, net, buffer_size, input_size):
        super(GDumb, self).__init__(net, buffer_size, input_size)
        self.memory_by_class = {}  # memory dict = {0: [ex0, ex1, ..., ex10], 1: [ex0, ex1], 2: []}
        self.memory_class_counter = {}  # class dict = {0: 10, 1: 2}  - counter for each class

    def update_buffer(self, x, y):
        batch_size = x.shape[0]
        for j in range(batch_size):
            x_j, y_j = x[j], y[j].item()
            # the memory must contain all the classes in a balanced way -> k_c samples for each class
            k_c = self.buffer_size // max(1, len(self.memory_by_class))  # k / |Y|
            if y_j not in self.memory_by_class or self.memory_class_counter[y_j] < k_c:
                if sum(self.memory_class_counter.values()) >= self.buffer_size:
                    # memory is full, replace -> Select largest class, break ties randomly
                    cls_max = max(self.memory_class_counter.items(), key=lambda k: k[1])[0]
                    idx = random.randrange(self.memory_class_counter[cls_max])
                    self.memory_by_class[cls_max].pop(idx)
                    self.memory_class_counter[cls_max] -= 1
                if y_j not in self.memory_by_class:
                    # init memory bank for new class
                    self.memory_by_class[y_j] = []
                    # init class counter for the new class
                    self.memory_class_counter[y_j] = 0
                # populate the memory by class
                self.memory_by_class[y_j].append(x_j)
                self.memory_class_counter[y_j] += 1

    def retrieve_buffer(self, num_retrieve=None, return_indices=False):
        return None, None

    def train_on_memory(self, batch_size, epochs, optimizer, clip=None):

        mem_x = torch.cat([torch.stack(self.memory_by_class[i]) for i in self.memory_by_class])
        mem_y = torch.LongTensor([i for i in self.memory_class_counter for j in range(self.memory_class_counter[i])],
                                 ).to(mem_x.device)

        # print(mem_y)

        self.net.apply(_reset_parameters)  # every task processed from scratch

        mem_ds = MemoryDataset(mem_x, mem_y)
        mem_dl = data.DataLoader(mem_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(epochs):
            pbar = tqdm(mem_dl)
            pbar.set_description(f"GDumb - Epoch {epoch + 1}/{epochs}")
            for x, y in pbar:
                optimizer.zero_grad()
                output = self.net(x)
                loss = F.cross_entropy(output, y, reduction='mean')
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()
                pbar.set_postfix({'Loss': loss.item()})
