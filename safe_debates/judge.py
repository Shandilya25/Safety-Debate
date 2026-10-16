import torch.nn as nn


class Judge(nn.Module):
    """
    Sparse MNIST classifier used as the debate judge.

    Adapted from the ``ConvNet`` in cle-mnist (https://github.com/jvmncs/cle-mnist), which
    in turn follows the TensorFlow CNN MNIST tutorial.  The only change from upstream is the
    number of input channels: channel 0 is the reveal mask, channel 1 the revealed pixel values.

    Note: like upstream, there is no padding (so the flattened size is 4*4*64) and no ReLU
    after ``hidden``.  Both differ from the TF tutorial but are kept so that existing
    checkpoints and the published judge accuracies remain valid.
    """
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(2, 32)
        self.block2 = self.conv_block(32, 64)
        self.hidden = nn.Linear(4 * 4 * 64, 1024)
        self.dropout = nn.Dropout(.4)
        self.out = nn.Linear(1024, 10)

    @classmethod
    def conv_block(cls, in_channels, out_channels):
        """A single convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )

    def forward(self, x):
        """Forward pass on (B, 2, 28, 28) mask/value inputs; returns (B, 10) logits."""
        x = x.view(-1, 2, 28, 28)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        x = self.dropout(self.hidden(x))
        return self.out(x)
