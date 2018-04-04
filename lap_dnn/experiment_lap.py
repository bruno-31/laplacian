from ray import tune
import ray
from train_lap_dnn import train_with_dic
import random
import os
import argparse
from ray.tune import Experiment, run_experiments
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from test import train

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # GPU available

tune.register_trainable("train_dnn", train_with_dic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--name", help="name_experiments",type=str, default="experiment")
    parser.add_argument(
        "--verbose", help="verbose experiments", default=False)
    parser.add_argument(
        "--repeat", help="repeat", type=int, default=10)
    args, _ = parser.parse_known_args()

    ray.init()
    exp = Experiment(
        name=args.name,
        run="train_dnn",
        trial_resources= {"cpu": 6, "gpu": 1},
        config={
            "verbose": args.verbose,
            "epoch": 200,
            "grad": True,
            "labeled": 400,
            "scale": lambda spec: 10 ** (random.uniform(-6,1)),  # log [10**-6 10**1]
            "reg_w": lambda spec: 10 ** (random.uniform(-5,2)),  # log [10**-5 10**2]
            "mc_size": lambda spec: random.randint(1, 12) * 10,  # [10**1 10**3]
            "batch_size": lambda spec: 25 + 25 * int(4 * random.random()), # [25 50 75 100]
            "learning_rate": lambda spec: 3* 10 ** (random.uniform(-5,-3))},  # logarithmic [3e-3 3e-5]
        local_dir="./ray_results",
        repeat=args.repeat,
        max_failures= 2,
        stop= {"training_iteration": 1 if args.smoke_test else 99999}
    )

    ahb = AsyncHyperBandScheduler(
        time_attr="timesteps_total", reward_attr="mean_accuracy", grace_period=40, max_t=200)

    run_experiments(exp, verbose=args.verbose, scheduler=ahb)

