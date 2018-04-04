
import argparse
import itertools

def train(config, reporter=None):
    global FLAGS, status_reporter, activation_fn
    status_reporter = reporter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir', type=str, default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS.datadir)
    print(unparsed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--name", help="name_experiments",type=str, default="experiment")

    args, x = parser.parse_known_args()
    # dict([s.split('--') for s in x.split()])
    d = dict(itertools.zip_longest(*[iter(x)] * 2, fillvalue=""))
    print(args)
    print(d)