class BatchDataGenerater:
    def __init__(self, args, src, tar, idx):
        self.args = args
        self.src = src
        self.tar = tar
        self.idx = idx
        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.iter_count < len(self.idx):
            src_inputs = self.src['inputs'][self.idx[self.iter_count: \
                    min(self.iter_count+self.args.batch_size, len(self.idx))]]
            src_pad_mask = self.src['pad_masks'][self.idx[self.iter_count: \
                    min(self.iter_count+self.args.batch_size, len(self.idx))]]
            tmp_tar_inputs = self.tar['inputs'][self.idx[self.iter_count: \
                    min(self.iter_count + self.args.batch_size, len(self.idx))]]
            tar_pad_mask = self.tar['pad_masks'][self.idx[self.iter_count: \
                    min(self.iter_count + self.args.batch_size, len(self.idx))]]

            tar_inputs = tmp_tar_inputs[:, :-1]
            tar_labels = tmp_tar_inputs[:, 1:]

            self.iter_count += self.args.batch_size

            # return src_inputs, src_pad_mask, tar_inputs, tar_pad_mask[:, :-1], tar_labels
            return src_inputs.cuda(), src_pad_mask.cuda(), tar_inputs.cuda(), tar_pad_mask[:, :-1].cuda(), tar_labels.cuda()

        else:
            self.iter_count = 0
            raise StopIteration
