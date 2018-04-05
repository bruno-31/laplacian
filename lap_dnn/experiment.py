from ray import tune
import ray
from train_lap_dnn import train_with_dic
import random
import os
import argparse
from ray.tune import Experiment, run_experiments
from ray.tune.async_hyperband import AsyncHyperBandScheduler
import itertools

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # GPU available

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--name", help="name_experiments",type=str, default="experiment")
    parser.add_argument(
        "--verbose", help="verbose experiments",default=True)
    parser.add_argument(
        "--repeat", help="repeat", type=int, default=1)
    parser.add_argument(
        "--max_failure", help="repeat", type=int, default=2)
    args, ukn_args = parser.parse_known_args()

    ukn_args = dict(itertools.zip_longest(*[iter(ukn_args)] * 2, fillvalue=""))

    # for attr, value in sorted(ukn_args.items()):
    #     print("{}={}".format(attr.lower(), value))
    # print("")

    # _config ={
    #         "verbose": args.verbose,
    #         "scale": lambda spec: 10 ** (random.uniform(-6,1)),  # log [10**-6 10**1]
    #         "reg_w": lambda spec: 10 ** (random.uniform(-5,2)),  # log [10**-5 10**2]
    #         "mc_size": lambda spec: random.randint(1, 12) * 10,  # [10**1 10**3]
    #         "batch_size": lambda spec: 25 + 25 * int(4 * random.random()), # [25 50 75 100]
    #         "learning_rate": lambda spec: 3* 10 ** (random.uniform(-5, -3))}  # logarithmic [3e-3 3e-5]

    # _config = {
    #     "labeled": 100,
    #     "verbose": args.verbose,
    #     "scale": tune.grid_search([1e-3, 1e-2, 5e-2 ,1e-1, 5e-1, 1.]),
    #     "reg_w": tune.grid_search([5e-3, 1e-3, 5e-4 ,1e-4]),  # log [10**-5 10**2]
    #     "mc_size": tune.grid_search([200, 100, 50])}

    _config = {"verbose": args.verbose,
               "labeled": tune.grid_search([10, 100, 200, 400]),
               "grad": tune.grid_search(['stochastic'])}

    config ={}
    for d in (_config, ukn_args): config.update(d)

    tune.register_trainable("train_dnn", train_with_dic)

    ray.init()
    exp = Experiment(
        name=args.name,
        run="train_dnn",
        trial_resources= {"cpu": 6, "gpu": 1},
        config=config,
        local_dir="./ray_results",
        repeat=args.repeat,
        max_failures= args.max_failure,
        stop={"training_iteration": 1 if args.smoke_test else 99999}
    )

    ahb = AsyncHyperBandScheduler(
        time_attr="timesteps_total", reward_attr="mean_accuracy", grace_period=40, max_t=200)

    # run_experiments(exp, verbose=args.verbose, scheduler=ahb)

    run_experiments(exp, verbose=args.verbose)
