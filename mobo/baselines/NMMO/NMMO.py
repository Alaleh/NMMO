from __future__ import annotations

import os
import time
import copy
import torch
import random
import numpy as np
from torch import optim
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from mobo.baselines.NMMO.acquisition import LowerBoundMC
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize, normalize
from mobo.baselines.NMMO.NMMO_utils import initialize_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mobo.utils.general_utils import write_to_file, generate_initial_data, get_test_problem
from botorch.utils.multi_objective.box_decompositions.dominated import (DominatedPartitioning, )
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

RAW_SAMPLES = 1024
NUM_RESTARTS = 20
MC_SAMPLES = 128
N_NM_SAMPLES = 10
GRID_SIZE = 200

def optimize_nmmo_and_get_observation(model, train_x, problem, sampler, horizon, nmmo_selection, tkwargs):
    """Optimizes the nmmo acquisition function, and returns a new candidate and observation."""

    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=problem.ref_point, Y=pred, )

    if "Det" in nmmo_selection:
        lookahead_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
    else:
        lookahead_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_NM_SAMPLES]))

    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1

    if "Joint" in nmmo_selection:
        acq_func = LowerBoundMC(
            model=model,
            ref_point=problem.ref_point,
            partitioning=partitioning,
            sampler=sampler,
            lookahead_sampler=lookahead_sampler,
            compute_approach=nmmo_selection)
        # optimize
        candidates, vals = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=horizon,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=False, )
        candidates = candidates[0].unsqueeze(0)

    elif "Nested" in nmmo_selection:
        acq_func = LowerBoundMC(model=model,
                                ref_point=problem.ref_point,
                                partitioning=partitioning,
                                sampler=sampler,
                                lookahead_sampler=lookahead_sampler,
                                compute_approach=nmmo_selection,
                                standard_bounds=standard_bounds,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES,
                                horizon_length=horizon).to(**tkwargs)

        grid = draw_sobol_samples(bounds=standard_bounds, n=GRID_SIZE, q=1).squeeze(-2)
        max_acqfuncvals = acq_func(grid[0].unsqueeze(0).unsqueeze(0))
        candidates = grid[0].unsqueeze(0)

        grid_opt = draw_sobol_samples(bounds=standard_bounds, n=6, q=1).squeeze(-2)
        for cur_x in grid_opt:
            cur_x_param = torch.nn.Parameter(cur_x)  # Wrap cur_x in a Parameter object
            cur_x_param.requires_grad = True

            optimizer_sgd = optim.SGD([cur_x_param], lr=0.04, maximize=True)  # Adjust learning rate as needed

            for qq in range(50):
                optimizer_sgd.zero_grad()
                objective = acq_func(cur_x_param.unsqueeze(0).unsqueeze(0))
                objective.backward()
                optimizer_sgd.step()
                cur_x_param.data = torch.clamp(cur_x_param.data, min=0.0, max=1.0)

                if objective.item() > max_acqfuncvals:
                    candidates = cur_x_param.clone().detach().unsqueeze(0)
                    max_acqfuncvals = objective.item()

    else:
        ValueError("Pick approximation approach")

    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    return new_x, new_obj_true


def run_NMMO(problem_name, n_var, n_obj, n_iter, n_init, res_path, seed, horizon, nmmo_selection, tkwargs):
    minimization = False

    problem = get_test_problem(problem_name, n_var, n_obj).to(**tkwargs)
    train_x, train_obj = generate_initial_data(problem, n_init)

    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
    volume = bd.compute_hypervolume().item()
    hvs = []
    hvs.append(volume)

    write_to_file(train_x, res_path + '/original_input.txt')
    write_to_file(train_obj, res_path + '/original_output.txt')
    write_to_file(torch.tensor([[volume]]), res_path + '/hv.txt')

    if "Det" in nmmo_selection:
        deterministic_check = True
    else:
        deterministic_check = False


    t0 = time.time()
    for iteration in range(n_init, n_iter + n_init):
        print("Starting NMMO seed " + str(seed) + " iteration " + str(iteration) + " with HV " + str(
            hvs[-1]) + " in " + str(time.time() - t0))
        t0 = time.time()


        # fit the models
        mll, model_nmmo = initialize_model(train_x, train_obj, problem, det=deterministic_check)
        fit_gpytorch_mll(mll)

        # define the acquisition modules using a QMC sampler
        nmmo_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # optimize acquisition functions and get new observations
        new_x, new_obj = optimize_nmmo_and_get_observation(model_nmmo, train_x, problem, nmmo_sampler,
                                                           horizon, nmmo_selection, tkwargs)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
        volume = bd.compute_hypervolume().item()
        hvs.append(volume)
        t1 = time.time()

        write_to_file(train_x[-1:], res_path + '/original_input.txt')
        write_to_file(train_obj[-1:], res_path + '/original_output.txt')
        write_to_file(torch.tensor([[volume]]), res_path + '/hv.txt')
        write_to_file(torch.tensor([[t1-t0]]), res_path + '/iteration_time.txt')


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='NMMO program for optimization')
    parser.add_argument('--problem', type=str, default='ZDT3', help='problem to test')
    parser.add_argument('--n_var', type=int, default=7, help='number of input dimensions')
    parser.add_argument('--n_obj', type=int, default=2, help='number of output dimensions')
    parser.add_argument('--n_init', type=int, default=0, help='number of initial BO points')
    parser.add_argument('--n_iter', type=int, default=100, help='number of BO iterations')
    parser.add_argument('--look_ahead_horizon', type=int, default=4, help='look-ahead horizon')
    parser.add_argument('--nmmo_method', type=str, default='LbJointDet', help='NMMO selection method')
    parser.add_argument('--seed', type=int, default=1, help='number of different seeds')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    res_path = 'results/NMMO/NMMO_results/NMMO_' + args.nmmo_method + '_horizon_' + str(args.look_ahead_horizon) + \
               '_results/' + args.problem + '(' + str(args.n_var) + ',' + str(args.n_obj) + ')' + '/experiment_' + \
               str(args.seed)
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
        open(res_path + '/original_input.txt', 'a').close()
        open(res_path + '/original_output.txt', 'a').close()
        open(res_path + '/hv.txt', 'a').close()
        open(res_path + '/iteration_time.txt', 'a').close()

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(
            "cuda:" + str(args.seed % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu"), }

    print("Using GPUs? ", torch.cuda.is_available())

    run_NMMO(args.problem, args.n_var, args.n_obj, args.n_iter, args.n_init, res_path, args.seed,
             args.look_ahead_horizon, args.nmmo_method, tkwargs)
