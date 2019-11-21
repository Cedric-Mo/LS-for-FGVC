import argparse


# define hyperparameter
def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['cubbirds', 'stdogs', 'stcars', 'vggaircraft', 'nabirds'],
                        help='choose between cubbirds, stdogs, stcars, vggaircraft, nabirds')
    parser.add_argument('--osme', '-o', action='store_true', help='osme module flag')
    parser.add_argument('--nparts', type=int, default=1,  help='number of parts')
    parser.add_argument('--gamma1', type=float, default=1.0, help='gamma1')
    parser.add_argument('--gamma2', type=float, default=0.65, help='gamma2')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--milestones', type=int, nargs='+', default=[16, 30], help='milestones')
    parser.add_argument('--device', type=str, default='0,1', help='device ID')

    args = parser.parse_args()

    return args
