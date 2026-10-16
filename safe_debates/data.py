import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

IMAGE_SIZE = 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def sparse_state(images, pixels):
    """
    Build judge inputs from images and revealed pixel indices.

    images: (B, 1, 28, 28) float; pixels: (B, k) long flat (row-major) pixel indices.
    Returns (B, 2, 28, 28): channel 0 is the reveal mask, channel 1 the revealed values.
    """
    flat = images.reshape(images.size(0), -1)
    mask = torch.zeros_like(flat).scatter_(1, pixels, 1.)
    return torch.stack((mask, flat * mask), dim=1).view(-1, 2, IMAGE_SIZE, IMAGE_SIZE)


class SparseMNIST(datasets.MNIST):
    """
    MNIST where each sample reveals ``pixels`` nonzero pixels chosen uniformly at random
    (without replacement).  ``n`` sets the epoch length; indices wrap around the dataset.

    Returns ``(sparse, img, target)`` with ``sparse`` = cat(mask, masked image) of shape
    (2, 28, 28).  Uses the same ``torch.multinomial`` call on the default generator as the
    original implementation, so a given seed yields the same pixel choices.
    """
    def __init__(self, pixels, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixels = pixels
        self._n = n
        self._generator = torch.default_generator

    @property
    def raw_folder(self):
        # torchvision derives the folder from the class name; share the plain MNIST download
        return os.path.join(self.root, 'MNIST', 'raw')

    def __getitem__(self, ix):
        ix = ix % len(self.data)
        img, target = self.data[ix], self.targets[ix]
        img = img.unsqueeze(0).type(torch.float32) / 255
        mask = self._get_mask(img)
        sparse = img * mask

        if self.transform is not None:
            img = self.transform(img)
            sparse = self.transform(sparse)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.cat((mask, sparse)), img, target

    def __len__(self):
        return self._n

    def _get_mask(self, img):
        mask = torch.zeros_like(img)
        mask.view(-1)[self._get_pixels(img)] = 1.
        return mask

    def _get_pixels(self, img):
        indices = (img.view(-1) > 0).nonzero().squeeze(1)
        choices = torch.multinomial(torch.ones(indices.size(0)), self.pixels,
                                    generator=self._generator)
        return indices.index_select(0, choices)


def prepare_data(args):
    """Prepare SparseMNIST DataLoaders for training/test and optional validation"""
    train_set = SparseMNIST(args.pixels, args.batches * args.batch_size, args.data_folder,
                            train=True, download=True)
    if args.evaluate:
        val_set = SparseMNIST(args.pixels, args.val_batches * args.eval_batch_size,
                              args.data_folder, train=True)
    test_set = SparseMNIST(args.pixels, args.test_batches * args.eval_batch_size,
                           args.data_folder, train=False)

    kwargs = {'num_workers': args.workers, 'pin_memory': args.use_cuda}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, **kwargs) \
        if args.evaluate else None
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, **kwargs)

    return train_loader, val_loader, test_loader
