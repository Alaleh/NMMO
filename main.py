import os
import torch
import signal
from time import time
from multiprocessing import Process, Queue
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def worker(cmd, problem, algo, seed, queue):
    ret_code = os.system(cmd)
    queue.put([ret_code, problem, algo, seed])


def main():

    look_ahead_baselines = ["BINOM", "NMMO"]

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Main program for optimization')
    parser.add_argument('--problem', type=str, default='FBTD', help='problem to test')
    parser.add_argument('--n_var', type=int, default=4, help='number of input dimensions')
    parser.add_argument('--n_obj', type=int, default=2, help='number of output dimensions')
    parser.add_argument('--algo', type=str, default='NMMO', help='algorithms to test')
    parser.add_argument('--n_iter', type=int, default=30, help='number of initial BO points')
    parser.add_argument('--n_init', type=int, default=5, help='number of BO iterations')
    parser.add_argument('--look_ahead_horizon', type=int, default=2, help='look-ahead horizon')
    parser.add_argument('--binom_method', type=str, default='max', help='BINOM selection method (sample/max)')
    parser.add_argument('--nmmo_method', type=str, default='LbJointDet', help='NMMO selection method')
    parser.add_argument('--start_seed', type=int, default=0, help='number of starting seeds')
    parser.add_argument('--n_seed', type=int, default=1, help='number of different seeds')
    args = parser.parse_args()

    start_time = time()

    if not torch.cuda.is_available():
        n_parallel_processes = os.cpu_count()//8 + 1
    else:
        n_parallel_processes = 2*torch.cuda.device_count()

    if n_parallel_processes <= 1:
        print("Parallelization error")
        n_parallel_processes = 1

    queue = Queue()
    n_active_process = 0

    for seed in range(args.start_seed, args.n_seed):
        command = f'python -W ignore -m mobo.baselines.{args.algo}.{args.algo} --problem {args.problem} --n_var {args.n_var} \
                  --n_obj {args.n_obj} --seed {seed} --n_iter {args.n_iter} --n_init {args.n_init}'

        if args.algo.startswith("BINOM"):
            command += f' --binom_method {args.binom_method}'

        if args.algo == "NMMO":
            command += f' --nmmo_method {args.nmmo_method}'

        if args.algo in look_ahead_baselines:
            command += f' --look_ahead_horizon {args.look_ahead_horizon}'

        Process(target=worker, args=(command, args.problem, args.algo, seed, queue)).start()
        print(f'problem {args.problem} algo {args.algo} seed {seed} started')
        n_active_process += 1

        if n_active_process >= n_parallel_processes:
            ret_code, ret_problem, ret_algo, ret_seed = queue.get()
            if ret_code == signal.SIGINT:
                exit()
            print(f'problem {ret_problem} algo {ret_algo} seed {ret_seed} done, time: ' + '%.2fs' % (
                    time() - start_time))
            n_active_process -= 1

    for _ in range(n_active_process):
        ret_code, ret_problem, ret_algo, ret_seed = queue.get()
        if ret_code == signal.SIGINT:
            exit()
        print(f'problem {ret_problem} algo {ret_algo} seed {ret_seed} done, time: ' + '%.2fs' % (time() - start_time))

    print('all experiments done, time: %.2fs' % (time() - start_time))


if __name__ == "__main__":
    main()
