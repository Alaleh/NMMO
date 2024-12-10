import csv
import copy
import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple, Union
from botorch.test_functions.base import MultiObjectiveTestProblem


def safe_div(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    if not torch.is_tensor(x1):
        x1 = torch.full_like(x2, fill_value=x1)
    c = torch.full_like(x1, fill_value=0.0, dtype=x2.dtype, device=x2.device)
    mask = (x2 != 0)
    c[mask] = x1[mask] / x2[mask]
    return c


def read_problem_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        tensor_list = [torch.tensor([float(item) for item in row], dtype=torch.double) for row in reader]
    return torch.stack(tensor_list)


class FBTD(MultiObjectiveTestProblem):
    '''
    Four bar truss design
    '''
    _ref_point = [2967.0243, 0.0383]
    _max_hv = None
    _bounds = [(1.0, 3.0), (np.sqrt(2), 3.0), (np.sqrt(2), 3.0), (1.0, 3.0)]
    dim = 4
    num_objectives = 2

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = torch.split(X, 1, -1)

        F = 10
        E = 2e5
        L = 200

        f1 = L * ((2 * x1) + np.sqrt(2.0) * x2 + torch.sqrt(x3) + x4)
        f2 = (F * L) / E * (safe_div(2.0, x1) + safe_div(2.0 * np.sqrt(2.0), x2) -
                            safe_div(2.0 * np.sqrt(2.0), x3) + safe_div(2.0, x4))
        f_X = torch.cat([f1, f2], dim=-1)

        return f_X


class RCBD(MultiObjectiveTestProblem):
    '''
    Reinforced Concrete Beam Design
    '''
    _ref_point = [703.6860, 899.2291]
    _max_hv = None
    _bounds = [(0.2, 15.0), (0.0, 20.0), (0.0, 40.0)]
    dim = 3
    num_objectives = 2

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3 = torch.split(X, 1, -1)

        closest_list = torch.tensor([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32,
                                     1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60,
                                     2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0,
                                     4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0,
                                     6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0,
                                     11.06, 11.85, 12.0, 13.0, 14.0, 15.0], dtype=x1.dtype).to(x1.device)

        # Find the closest values from the list to values in x1
        min_indices = torch.argmin(torch.abs(closest_list - x1), dim=1)

        x1 = closest_list[min_indices].reshape(x1.size())

        f1 = (29.4 * x1) + (0.6 * x2 * x3)

        g = torch.column_stack([(x1 * x3) - 7.735 * safe_div((x1 * x1), x2) - 180.0, 4.0 - safe_div(x3, x2)])

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f2 = torch.sum(g, dim=1).reshape(-1, 1)
        f_X = torch.stack([f1, f2], dim=-1).squeeze(1)

        return f_X


class WBD(MultiObjectiveTestProblem):
    '''
    Welded Beam Design
    '''
    _ref_point = [202.8569, 42.0653, 2111643.6209]
    _max_hv = None
    _bounds = [(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.125, 5.0)]
    dim = 4
    num_objectives = 3

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = torch.split(X, 1, -1)

        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f2 = safe_div(4 * P * L * L * L, E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = safe_div(M * R, J)
        tauDash = safe_div(P, np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + safe_div((2 * tauDash * tauDashDash * x2), (2 * R)) + (tauDashDash * tauDashDash)
        tau = torch.sqrt(tmpVar)
        sigma = safe_div(6 * P * L, x4 * x3 * x3)
        tmpVar = 4.013 * E * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g = torch.column_stack([tauMax - tau, sigmaMax - sigma, x4 - x1, PC - P])

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f3 = torch.sum(g, dim=1).reshape(-1, 1)
        f_X = torch.stack([f1, f2, f3], dim=-1).squeeze(1)

        return f_X


class DBD(MultiObjectiveTestProblem):
    '''
    Disc Brake Design
    '''
    _ref_point = [6.1356, 6.3421, 12.9737]
    _max_hv = None
    _bounds = [(55.0, 80.0), (75.0, 110.0), (1000.0, 3000.0), (11.0, 20.0)]
    dim = 4
    num_objectives = 3

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = torch.split(X, 1, -1)

        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        f2 = safe_div((9.82 * 1e6) * (x2 * x2 - x1 * x1), x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        t1 = (x2 - x1) - 20.0
        t2 = 0.4 - safe_div(x3, (3.14 * (x2 * x2 - x1 * x1)))
        t3 = 1.0 - safe_div(2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1), torch.pow((x2 * x2 - x1 * x1), 2))
        t4 = safe_div(2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1), x2 * x2 - x1 * x1) - 900.0

        g = torch.column_stack([t1, t2, t3, t4])

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f3 = torch.sum(g, dim=1).reshape(-1, 1)
        f_X = torch.stack([f1, f2, f3], dim=-1).squeeze(1)

        return f_X


class GTD(MultiObjectiveTestProblem):
    '''
    Gear train design
    '''

    _ref_point = [6.6764, 59.0, 0.4633]
    _max_hv = None
    _bounds = [(12.0, 60.0)] * 4
    dim = 4
    num_objectives = 3

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
        x1, x2, x3, x4 = torch.round(x1), torch.round(x2), torch.round(x3), torch.round(x4)

        f1 = torch.abs(6.931 - (safe_div(x3, x1) * safe_div(x4, x2)))
        f2, _ = torch.max(torch.column_stack([x1, x2, x3, x4]), dim=1)

        g = 0.5 - (f1 / 6.931)
        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]
        f3 = g

        f_X = torch.stack([f1, f2, f3], dim=-1)

        return f_X


class MOF(MultiObjectiveTestProblem):
    '''
    metal organic frameworks
    '''
    _ref_point = [1.0, 1.0]
    _max_hv = None
    _bounds = [(0.0,5.2), (0.0, 9750.1), (0.0, 3995.1), (0.0, 1.0), (0.0, 35.74), (0.0, 71.6), (0.0, 71.6)]
    dim = 7
    num_objectives = 2

    def __init__(self, noise_std: Union[None, float, List[float]] = None, negate: bool = False,) -> None:
        _ref_point = [1.0, 1.0]
        _max_hv = None
        _bounds = [(0.0, 5.2), (0.0, 9750.1), (0.0, 3995.1), (0.0, 1.0), (0.0, 35.74), (0.0, 71.6), (0.0, 71.6)]
        dim = 7
        num_objectives = 2

        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(noise_std=noise_std, negate=negate)
        ref_point = torch.tensor(self._ref_point, dtype=torch.float)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

        self.bound = [(0.0,5.2), (0.0, 9750.1), (0.0, 3995.1), (0.0, 1.0), (0.0, 35.74), (0.0, 71.6), (0.0, 71.6)]
        self.MOF_input_space = read_problem_data('mobo/utils/benchmark_data/MOF/MOF_input.csv')
        self.MOF_output_space = read_problem_data('mobo/utils/benchmark_data/MOF/MOF_output.csv')

        for di in range(self.MOF_input_space.size(-1)):
            lower, upper = _bounds[di]
            self.MOF_input_space[:, di] = (self.MOF_input_space[:, di] - lower) / (upper - lower)

    def evaluate_true(self, X: Tensor) -> Tensor:

        out = []

        for x in X:
            min_dist = float('inf')
            min_index = 0

            norm_x = torch.zeros_like(x)
            for i, (lower, upper) in enumerate(self.bound):
                norm_x[i] = (x[i] - lower) / (upper - lower)

            for i, tensor in enumerate(self.MOF_input_space):
                # Compute the Euclidean distance))
                dist = torch.norm(norm_x - tensor.to(norm_x.device))

                # Update the minimum distance and index if a new minimum is found
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            out.append(-copy.deepcopy(self.MOF_output_space[min_index]).to(x.device))

        return torch.stack(out)