from ray import tune
import ray
from cnn import train_with_dic
import random
import os
import argparse
from ray.tune import Experiment, run_experiments
from ray.tune.async_hyperband import AsyncHyperBandScheduler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # GPU available

tune.register_trainable("train_dnn", train_with_dic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()

    exp = Experiment(
        name="my_test",
        run="train_dnn",
        trial_resources= {"cpu": 6, "gpu": 1},
        config={
            "epoch": 20,
            "labeled":100,
            "batch_size": lambda spec: 25 + 25 * int(4 * random.random()),
            "learning_rate": lambda spec: 10 ** (2 * random.random() - 5)},  # logarithmic 10^[-5,-3]
        local_dir= "./ray_results",
        repeat= 20,
        max_failures= 3,
        stop= {"training_iteration": 1 if args.smoke_test else 99999}
    )

    ahb = AsyncHyperBandScheduler(
        time_attr="timesteps_total", reward_attr="mean_accuracy",
        grace_period=5, max_t=100)

    run_experiments(exp, verbose=False,scheduler=ahb)

