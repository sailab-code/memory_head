import torch
from competitors import CompetitorWrapperER
import numpy as np


class ERReservoir(CompetitorWrapperER):
    def __init__(self, net, buffer_size, input_size):
        super(ERReservoir, self).__init__(net, buffer_size, input_size)

        # registering as buffer - number of instances encountered so far
        # self.register_buffer('n', torch.zeros(1))
        # registering as buffers the sample buffer and the corresponding target buffer
        self.register_buffer('sample_buffer', torch.zeros(buffer_size, *input_size))
        self.register_buffer('label_buffer', torch.zeros(buffer_size, dtype=torch.long))

        self.current_index = 0
        self.n_seen_so_far = 0
        # label_buffer
        # sample_buffer
        # buffer_size

    def update_buffer(self, x, y):

        """
        During the first phase, which lasts until the memory gets filled, all encountered data instances are stored
        in empty memory spots. In the second phase, which starts once the memory gets filled and continues from
        then on, the currently observed data instance is stored with probability m/n, where m is the size of the memory
        and n is the number of data instances encountered so far. The data instance is stored in a memory spot that is
        uniformly selected, thus all currently stored instances are equally likely to be overwritten.
        Also : https://arxiv.org/pdf/1902.10486.pdf
        """
        # list of filled idx
        filled_idx = []
        batch_size = x.size(0)
        # for cleaner code
        c_idx = self.current_index
        # add whatever still fits in the buffer
        available_slots = max(0, self.buffer_size - c_idx)

        # there are still available slots to be filled in the memory
        if available_slots:
            offset = min(available_slots, batch_size)
            self.sample_buffer[c_idx: c_idx + offset].data.copy_(x[:offset])
            self.label_buffer[c_idx: c_idx + offset].data.copy_(y[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            filled_idx = list(range(self.current_index - offset, self.current_index))
            # everything coming from outside was added to memory, we are done!
            if offset == batch_size:
                return filled_idx

        # there are no more available slots - we should scramble some
        # maybe we have already filled some data into buffer, avoid to reuse it (batch case)
        x, y = x[available_slots:], y[available_slots:]
        remaining_x = x.size(0)
        # for every element of the (remaining) batch decide to store it or not
        # randint(0, n + j) for every element of the (remaining) batch
        destination_indices = torch.randint(low=0, high=self.n_seen_so_far, size=(remaining_x,), dtype=torch.int64,
                                            device=x.device)
        #  if i < buffer_size then they are valid indices -> we store it
        # it is the same as storing with  probability m/n, where m is the size of the memory and
        # n is the number of data instances encountered so far.

        # binary mask of the same size of the remaining batch  [0,1,0,1,0,1]  => 0 not valid idx, 1 valid idx

        mask_valid_destination_idx = destination_indices < self.buffer_size  # boolean mask
        # get batch idx corresponding to data that fulfills the condition => we will store this data!
        idx_new_data = mask_valid_destination_idx.nonzero().squeeze(-1)  # get idx of non zero elements
        # select only valid destination indices
        buffer_idx_to_be_filled = destination_indices[mask_valid_destination_idx]

        self.n_seen_so_far += remaining_x

        # not scrambling, buffer is kept as it is
        if buffer_idx_to_be_filled.numel() == 0:
            return []

        assert buffer_idx_to_be_filled.max() < self.buffer_size
        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        # map buffer_index_to_be_filled:
        idx_map = {buffer_idx_to_be_filled[i].item(): idx_new_data[i].item() for i in
                   range(buffer_idx_to_be_filled.size(0))}

        # scramble data in selected buffer positions
        with torch.no_grad():
            self.sample_buffer[buffer_idx_to_be_filled] = x[idx_new_data]
            self.label_buffer[buffer_idx_to_be_filled] = y[idx_new_data]

        additional_filled_idx = list(idx_map.keys())
        return filled_idx + additional_filled_idx


class ERRandom(ERReservoir):
    def __init__(self, net, buffer_size, input_size):
        super(ERRandom, self).__init__(net, buffer_size, input_size)

    def update_buffer(self, x, y):
        # list of filled idx
        filled_idx = []
        batch_size = x.size(0)
        # for cleaner code
        c_idx = self.current_index
        # add whatever still fits in the buffer
        available_slots = max(0, self.buffer_size - c_idx)

        # there are still available slots to be filled in the memory
        if available_slots:
            offset = min(available_slots, batch_size)
            self.sample_buffer[c_idx: c_idx + offset].data.copy_(x[:offset])
            self.label_buffer[c_idx: c_idx + offset].data.copy_(y[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            filled_idx = list(range(self.current_index - offset, self.current_index))
            # everything coming from outside was added to memory, we are done!
            if offset == x.size(0):
                return filled_idx

        # there are no more available slots - we should scramble some
        # maybe we have already filled some data into buffer, avoid to reuse it (batch case)
        x, y = x[available_slots:], y[available_slots:]
        remaining_x = x.size(0)
        # for every element of the (remaining) batch decide to store it or not
        destination_indices = torch.randint(low=0, high=2, size=(remaining_x,), device=x.device)
        # we store it if probability is randint is 1
        mask_valid_destination_idx = torch.eq(destination_indices, 1)  # boolean mask
        # get batch idx corresponding to data that fulfills the condition => we will store this data!
        idx_new_data = mask_valid_destination_idx.nonzero().squeeze(-1)  # get idx of non zero elements
        # select  destination indices randomly sampled

        # Normalize input tensor to probabilities
        # probs = memory_indices / memory_indices.sum()
        num_samples = torch.numel(idx_new_data)
        if num_samples > 0:
            buffer_idx_to_be_filled = torch.multinomial(torch.ones(self.buffer_size, device=x.device).float(),
                                                        num_samples, replacement=False)

        else:
            # not scrambling, buffer is kept as it is
            return []

        assert buffer_idx_to_be_filled.max() < self.buffer_size
        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        # map buffer_index_to_be_filled:
        idx_map = {buffer_idx_to_be_filled[i].item(): idx_new_data[i].item() for i in
                   range(buffer_idx_to_be_filled.size(0))}

        # scramble data in selected buffer positions
        with torch.no_grad():
            # print(buffer_idx_to_be_filled.item())
            self.sample_buffer[buffer_idx_to_be_filled] = x[idx_new_data]
            self.label_buffer[buffer_idx_to_be_filled] = y[idx_new_data]

        additional_filled_idx = list(idx_map.keys())
        return filled_idx + additional_filled_idx
