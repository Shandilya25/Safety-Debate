"""Train the sparse MNIST judge (adapted from cle-mnist's main.py)."""
import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from safe_debates.data import prepare_data
from safe_debates.judge import Judge
from safe_debates.utils import mkdir_p, save_checkpoint, seed_everything, select_device


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean cross-entropy and number of correct predictions over ``loader``."""
    model.eval()
    loss, correct, num = 0., 0, 0
    for sparse, _, label in loader:
        sparse, label = sparse.to(device), label.to(device)
        logits = model(sparse)
        loss += F.cross_entropy(logits, label, reduction='sum').item()
        correct += logits.argmax(1).eq(label).sum().item()
        num += label.numel()
    return loss / num, correct, num


def main(args):
    seed_everything(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if args.checkpoint_filename is None:
        checkpoint_file = os.path.join(args.checkpoint, str(datetime.now())[:-10].replace(':', '-'))
    else:
        checkpoint_file = os.path.join(args.checkpoint, args.checkpoint_filename)

    args.use_cuda, device = select_device(args.no_cuda)
    args.evaluate = args.val_batches > 0
    train_loader, val_loader, test_loader = prepare_data(args)

    model = Judge().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('\n================= TRAINING =================')
    model.train()
    correct = 0
    train_num = args.batches * args.batch_size
    time0 = time.time()

    for ix, (sparse, img, label) in enumerate(train_loader):
        sparse, label = sparse.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(sparse)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()

        correct += logits.detach().argmax(1).eq(label).sum()
        if ix % args.log_interval == 0:
            seen = (ix + 1) * len(sparse)
            print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%\t'
                  'Time: {:.0f} min, {:.2f} s'.format(
                      seen, train_num, 100. * ix / len(train_loader), loss.item(),
                      100. * correct.item() / seen,
                      (time.time() - time0) // 60, (time.time() - time0) % 60))

    correct = int(correct)
    print('Train Accuracy: {}/{} ({:.2f}%)\tTrain Time: {:.0f} minutes, {:.2f} seconds\n'.format(
        correct, train_num, 100. * correct / train_num,
        (time.time() - time0) // 60, (time.time() - time0) % 60))

    state = {'state_dict': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    if args.evaluate:
        print('\n================== VALIDATION ==================')
        val_loss, val_correct, val_num = evaluate(model, val_loader, device)
        val_acc = 100. * val_correct / val_num
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, val_correct, val_num, val_acc))
        state.update(val_loss=val_loss, val_acc=val_acc)

    save_checkpoint(state, True, checkpoint_file)

    print('\n================== TESTING ==================')
    check = torch.load(checkpoint_file + '-best.pth.tar', map_location=device, weights_only=True)
    model.load_state_dict(check['state_dict'])
    test_loss, test_correct, test_num = evaluate(model, test_loader, device)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_correct, test_num, 100. * test_correct / test_num))

    print('Final model stored at "{}".'.format(checkpoint_file + '-best.pth.tar'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train the sparse MNIST debate judge')
    parser.add_argument('--pixels', type=int, required=True, metavar='C',
                        help='number of nonzero pixels to remain in sparse image')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--eval-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for evaluation (default: 1,000)')
    parser.add_argument('--batches', type=int, default=30000, metavar='N',
                        help='number of batches to train on (default: 30,000)')
    parser.add_argument('--val-batches', type=int, default=4, metavar='N',
                        help='batches to use for validation (default: 4)')
    parser.add_argument('--test-batches', type=int, default=800, metavar='N',
                        help='batches to use for testing (default: 800)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='DataLoader worker processes (default: 1; changing it changes '
                             'the sampled pixels for a given seed)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log training status every N batches (default: 1000)')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='PATH',
                        help='root path for folder containing MNIST data download \
                        (default: ./data/)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', metavar='PATH',
                        help='root path for folder containing model checkpoints \
                        (default: ./checkpoint/)')
    parser.add_argument('--checkpoint-filename', type=str, default=None, metavar='PATH',
                        help='filename for model checkpoint (default: current datetime)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
