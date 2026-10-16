"""Play MNIST debates between an honest agent and a liar in front of a trained judge."""
import argparse
import cProfile
import pstats
import time

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from safe_debates.agent import Agent
from safe_debates.debate import Debate
from safe_debates.judge import Judge
from safe_debates.metrics import AverageMeter, track_stats_
from safe_debates.utils import load_judge_state, seed_everything, select_device


def main(args):
    seed_everything(args.seed)
    args.use_cuda, args.device = select_device(args.no_cuda)

    # data
    dataset = MNIST(args.data_folder, train=False, transform=ToTensor(), download=True)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=1,
                             pin_memory=args.use_cuda)
    if args.rounds is None:
        args.rounds = len(dataset) // args.batch_size

    # debate game
    judge = Judge().to(args.device)
    judge.load_state_dict(load_judge_state(args.checkpoint, args.device))
    judge.eval()
    helper = Agent(honest=True, args=args)
    liar = Agent(honest=False, args=args)
    debate = Debate((helper, liar), data_loader, args)

    total_meter = AverageMeter()
    class_meters = [AverageMeter() for _ in range(10)]

    for round_ in range(args.rounds):
        start = time.time()
        helper.precommit_(None, None)
        liar.precommit_(None, None)
        result = debate.play(judge, args.device)
        track_stats_(total_meter, class_meters, result['helper']['preds'],
                     result['helper']['wins'], result['labels'], args.precommit)
        print('round {}/{}: honest win rate {:.4f} (running {:.4f}), {:.1f}s'.format(
            round_ + 1, args.rounds, result['helper']['wins'].mean().item(),
            total_meter.avg, time.time() - start))

    print('Total accuracy: {}'.format(total_meter.avg))
    print('Accuracy per class\n==============================================')
    for i in range(10):
        print('Digit {}: {}'.format(i, class_meters[i].avg))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MNIST Debate Game')
    parser.add_argument('--pixels', type=int, required=True, metavar='C',
                        help='number of nonzero pixels to uncover in sparse image (required)')
    parser.add_argument('--checkpoint', type=str, required=True, metavar='CHAR',
                        help='model checkpoint to use as judge (required)')
    parser.add_argument('--rounds', type=int, default=None, metavar='N',
                        help='number of vectorized rounds (default: floor(10000 / batch_size))')
    parser.add_argument('--rollouts', type=int, default=10000, metavar='N',
                        help='number of rollouts to do before move selection (default: 10,000)')
    parser.add_argument('--c-puct', type=float, default=1., metavar='C',
                        help='PUCT exploration constant (default: 1.0)')
    parser.add_argument('--no-precommit', action='store_true', default=False,
                        help='playout games without precommit (currently required)')
    parser.add_argument('--honest-first', action='store_true', default=False,
                        help='honest player plays first')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='simultaneous games to play (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--data-folder', type=str, default='./data/', metavar='PATH',
                        help='root path for folder containing MNIST data download \
                        (default: ./data/)')
    parser.add_argument('--progress', action='store_true', default=False,
                        help='show a progress bar over rollouts')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='write cProfile stats to ./profiles/')
    args = parser.parse_args(argv)
    args.precommit = not args.no_precommit
    args.liar_first = not args.honest_first
    if args.precommit:
        parser.error('precommit debates are not implemented yet; pass --no-precommit')
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.profile:
        main(args)
    else:
        fname = './profiles/profile.{}px.{}rnds.{}roll.{}bs.{}.txt'.format(
            args.pixels, args.rounds, args.rollouts, args.batch_size,
            'cuda' if not args.no_cuda else 'nocuda')
        pr = cProfile.Profile()
        pr.enable()
        main(args)
        pr.disable()
        with open(fname, 'w') as f:
            pstats.Stats(pr, stream=f).sort_stats('cumulative').print_stats()
