from __future__ import annotations

import os
import time
import torch
import random
import numpy as np
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.model_list_gp_regression import ModelListGP
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from mobo.utils.general_utils import write_to_file, generate_initial_data, get_test_problem
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

NUM_RESTARTS = 20
RAW_SAMPLES = 1024
MC_SAMPLES = 128


def initialize_model(train_x, train_obj, problem):
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i: i + 1]
        models.append(SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_binom_and_get_observation(problem, model, train_x, sampler, horizon, selection_criteria, tkwargs):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1

    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=problem.ref_point, Y=pred, )

    acq_func = qExpectedHypervolumeImprovement(model=model, ref_point=problem.ref_point, partitioning=partitioning,
                                               sampler=sampler, ).to(**tkwargs)
    candidates, _ = optimize_acqf(acq_function=acq_func, bounds=standard_bounds, q=horizon, num_restarts=NUM_RESTARTS,
                                  raw_samples=RAW_SAMPLES, options={"batch_limit": 5, "maxiter": 200})
    one_step_acq_func = ExpectedHypervolumeImprovement(model=model, ref_point=problem.ref_point.tolist(),
                                                       partitioning=partitioning, ).to(**tkwargs)
    ehvi_values = one_step_acq_func(candidates.unsqueeze(1)).to(**tkwargs)

    if selection_criteria == "max":
        idx = torch.argmax(ehvi_values)
    elif selection_criteria == "sample":
        prob = ehvi_values.cpu().detach().numpy().squeeze()
        if min(prob) < 0:
            prob = [q - min(prob) for q in prob]
        prob = prob / np.sum(prob)
        idx = np.random.choice(np.arange(horizon), p=prob, replace=False)
    else:
        raise "Undefined selection method for BINOM!"
    candidates = candidates[idx].unsqueeze(0)
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    return new_x, new_obj_true


def run_BINOM(problem_name, n_var, n_obj, n_iter, n_init, res_path, seed, horizon, binom_selection, tkwargs):

    problem = get_test_problem(problem_name, n_var, n_obj).to(**tkwargs)
    train_x, train_obj = generate_initial_data(problem, n_init)

    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
    volume = bd.compute_hypervolume().item()
    hvs = []
    hvs.append(volume)

    write_to_file(train_x, res_path + '/original_input.txt')
    write_to_file(train_obj, res_path + '/original_output.txt')
    write_to_file(torch.tensor([[volume]]), res_path + '/hv.txt')

    # run N_BATCH rounds of BayesOpt after the initial random batch
    t0 = time.time()
    for iteration in range(n_init, n_iter + n_init):
        print("Starting BINOM seed " + str(seed) + " iteration " + str(iteration) + " with HV " + str(
            hvs[-1]) + " in " + str(time.time() - t0))
        t0 = time.time()

        # fit the models
        mll_binom, model_binom = initialize_model(train_x, train_obj, problem)
        fit_gpytorch_mll(mll_binom)

        binom_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # optimize acquisition functions and get new observations
        new_x_binom, new_obj_binom = optimize_binom_and_get_observation(problem, model_binom, train_x, binom_sampler,
                                                                        horizon, binom_selection, tkwargs)
        train_x = torch.cat([train_x, new_x_binom])
        train_obj = torch.cat([train_obj, new_obj_binom])

        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
        volume = bd.compute_hypervolume().item()
        hvs.append(volume)
        t1 = time.time()

        write_to_file(train_x[-1:], res_path + '/original_input.txt')
        write_to_file(train_obj[-1:], res_path + '/original_output.txt')
        write_to_file(torch.tensor([[volume]]), res_path + '/hv.txt')
        write_to_file(torch.tensor([[t1-t0]]), res_path + '/iteration_time.txt')


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='BINOM program for optimization')
    parser.add_argument('--problem', type=str, default='FBTD', help='problem to test')
    parser.add_argument('--n_var', type=int, default=4, help='number of input dimensions')
    parser.add_argument('--n_obj', type=int, default=2, help='number of output dimensions')
    parser.add_argument('--n_init', type=int, default=5, help='number of initial BO points')
    parser.add_argument('--n_iter', type=int, default=100, help='number of BO iterations')
    parser.add_argument('--look_ahead_horizon', type=int, default=4, help='look-ahead horizon')
    parser.add_argument('--binom_method', type=str, default='max', help='BINOM selection method (sample/max)')
    parser.add_argument('--seed', type=int, default=10, help='number of different seeds')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    res_path = 'results/BINOM/BINOM_' + args.binom_method + '_horizon_' + str(
        args.look_ahead_horizon) + '_results/' + args.problem + \
               '(' + str(args.n_var) + ',' + str(args.n_obj) + ')' + '/experiment_' + str(args.seed)

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

    run_BINOM(args.problem, args.n_var, args.n_obj, args.n_iter, args.n_init, res_path, args.seed,
              args.look_ahead_horizon, args.binom_method, tkwargs)
