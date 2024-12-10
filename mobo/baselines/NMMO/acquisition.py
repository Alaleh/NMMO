from __future__ import annotations

import sys
import torch
from torch import Tensor
from itertools import combinations
from botorch.models.model import Model
from botorch.utils.torch import BufferDict
from botorch.sampling.base import MCSampler
from typing import Any, Callable, List, Optional, Union
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.hypervolume import (NoisyExpectedHypervolumeMixin, SubsetIndexCachingMixin, )
from botorch.acquisition.multi_objective.objective import (IdentityMCMultiOutputObjective, MCMultiOutputObjective, )
from botorch.utils.transforms import concatenate_pending_points, is_fully_bayesian, match_batch_shape, \
    t_batch_mode_transform
from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction, \
    qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning, \
    NondominatedPartitioning


class AveragedqExpectedHypervolumeImprovement(MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin):
    def __init__(self,
            model: Model,
            ref_point: Union[List[float], Tensor],
            partitioning: NondominatedPartitioning,
            added_x: Tensor,
            lookahead_sampler: MCSampler,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCMultiOutputObjective] = None,
            constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
            X_pending: Optional[Tensor] = None,
            eta: Optional[Union[Tensor, float]] = 1e-3,
            fat: bool = False,) -> None:
        r"""q-Expected Hypervolume Improvement supporting m>=2 outcomes.
        See [Daulton2020qehvi]_ for details.
        """

        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError("The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected " f"{partitioning.num_outcomes}.")

        ref_point = torch.as_tensor(ref_point, dtype=partitioning.pareto_Y.dtype, device=partitioning.pareto_Y.device,)

        super().__init__(model=model, sampler=sampler, objective=objective, constraints=constraints,
                         eta=eta, X_pending=X_pending,)

        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        SubsetIndexCachingMixin.__init__(self)
        self.fat = fat
        self.added_x = added_x
        self.lookahead_sampler = lookahead_sampler
        self.fantasy_model = self.model.fantasize(X=self.added_x, sampler=self.lookahead_sampler)

    def _compute_qehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = compute_smoothed_feasibility_indicator(
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
                fat=self.fat,
            )  # `sample_shape x batch-shape x q`
        device = self.ref_point.device
        q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=device,)
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size([
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],])
        view_shape = (*sample_batch_view_shape, self.cell_upper_bounds.shape[-2], 1, self.cell_upper_bounds.shape[-1],)
        for i in range(1, self.q_out + 1):
            q_choose_i = q_subset_indices[f"q_choose_{i}"]
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:])
            overlap_vertices = obj_subsets.min(dim=-2).values
            overlap_vertices = torch.min(overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape))
            lengths_i = (overlap_vertices - self.cell_lower_bounds.view(view_shape)).clamp_min(0.0)
            areas_i = lengths_i.prod(dim=-1)
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            areas_per_segment += (-1) ** (i + 1) * areas_i
        return areas_per_segment.sum(dim=-1).mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.fantasy_model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        ehvi = self._compute_qehvi(samples=samples, X=X)
        ehvi = ehvi.mean(dim=0)
        return ehvi



class AveragedqExpectedHypervolumeImprovementPartitioning(MultiObjectiveMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        # partitioning: NondominatedPartitioning,
        added_x: Tensor,
        lookahead_sampler: MCSampler,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
    ) -> None:
        r"""modified qehvi to computed averaged one step lookahead version


        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.register_buffer("ref_point", ref_point)

        self.q_out = -1
        self.q_subset_indices = BufferDict()
        self.fantasy_model = self.model.fantasize(X=added_x, sampler=lookahead_sampler)
        with torch.no_grad():
            preds = self.fantasy_model.posterior(self.fantasy_model.train_inputs[0][0][0]).mean
        self.cell_lower_bounds_list=[]
        self.cell_upper_bounds_list=[]
        for pred in preds:
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point,
                Y=pred.squeeze(0),
            )
            cell_bounds = partitioning.get_hypercell_bounds()
            self.cell_lower_bounds_list.append(cell_bounds[0])
            self.cell_upper_bounds_list.append(cell_bounds[1])


    def _cache_q_subset_indices(self, q_out: int) -> None:
        r"""Cache indices corresponding to all subsets of `q_out`.

        This means that consecutive calls to `forward` with the same
        `q_out` will not recompute the indices for all (2^q_out - 1) subsets.

        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).

        Args:
            q_out: The batch size of the objectives. This is typically equal
                to the q-batch size of `X`. However, if using a set valued
                objective (e.g., MVaR) that produces `s` objective values for
                each point on the q-batch of `X`, we need to properly account
                for each objective while calculating the hypervolume contributions
                by using `q_out = q * s`.
        """
        if q_out != self.q_out:
            indices = list(range(q_out))
            tkwargs = {"dtype": torch.long, "device": self.ref_point.device}
            self.q_subset_indices = BufferDict(
                {
                    f"q_choose_{i}": torch.tensor(
                        list(combinations(indices, i)), **tkwargs
                    )
                    for i in range(1, q_out + 1)
                }
            )
            self.q_out = q_out

    def _compute_qehvi(self, samples: Tensor, cell_lower_bounds: Tensor, cell_upper_bounds: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = compute_smoothed_feasibility_indicator(
                constraints=self.constraints, samples=samples, eta=self.eta
            )  # `sample_shape x batch-shape x q`
        self._cache_q_subset_indices(q_out=q)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=obj.device,
        )
        cell_batch_ndim = cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            cell_upper_bounds.shape[-2],
            1,
            cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = self.q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:])
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(overlap_vertices.unsqueeze(-3), cell_upper_bounds.view(view_shape))
            # substract cell lower bounds, clamp min at zero
            lengths_i = (overlap_vertices - cell_lower_bounds.view(view_shape)).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1).mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.fantasy_model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        ehvi = []
        for s in range(samples.shape[1]):
            # print(self.cell_lower_bounds_list[s][0])
            ehvi.append(self._compute_qehvi(samples=samples[:, s, :, :],
                                                cell_lower_bounds=self.cell_lower_bounds_list[s],
                                                cell_upper_bounds=self.cell_upper_bounds_list[s], X=X).unsqueeze(0))
        ehvi = torch.cat(ehvi, dim=0)
        ehvi = ehvi.mean(dim=0)
        return ehvi


class LowerBoundMC(MultiObjectiveMCAcquisitionFunction):

    def __init__(self,
                 model: Model,
                 ref_point: Union[List[float], Tensor],
                 partitioning: NondominatedPartitioning,
                 lookahead_sampler: MCSampler,
                 compute_approach: str,
                 sampler: Optional[MCSampler] = None,
                 objective: Optional[MCMultiOutputObjective] = None,
                 constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
                 X_pending: Optional[Tensor] = None,
                 eta: Optional[Union[Tensor, float]] = 1e-3,
                 standard_bounds: Optional[Tensor] = None,
                 num_restarts: Optional[int] = None,
                 raw_samples: Optional[int] = None,
                 horizon_length: Optional[int] = None) -> None:

        super().__init__(model=model, sampler=sampler, objective=objective, constraints=constraints,
            eta=eta, X_pending=X_pending, )

        self.lookahead_sampler = lookahead_sampler
        self.partitioning = partitioning
        self.ref_point = ref_point
        self.compute_approach = compute_approach
        if compute_approach in ["LbNestedMCv0", "LbNestedMCv1", "LbJointDet", "LbNestedDet", "LbJointDetPartitioned"]:
            self.standard_bounds = standard_bounds
            self.num_restarts = num_restarts
            self.raw_samples = raw_samples
            self.horizon_length = horizon_length

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        first_step_acq_func = ExpectedHypervolumeImprovement(model=self.model, ref_point=self.ref_point.tolist(),
                                                             partitioning=self.partitioning, )
        first_step_val = first_step_acq_func(X[:, 0, :].unsqueeze(1))

        if self.compute_approach in ["LbNestedMCv0", "LbJointMCv0", "LbJointDet"]:
            average_ehvi = AveragedqExpectedHypervolumeImprovement(
                model=self.model,
                ref_point=self.ref_point,
                partitioning=self.partitioning,
                sampler=self.sampler,
                added_x=X[:, 0, :].unsqueeze(1),
                lookahead_sampler=self.lookahead_sampler, )
        elif self.compute_approach in ["LbNestedMCv1", "LbJointMCv1", "LbNestedDet", "LbJointDetPartitioned"]:
            average_ehvi = AveragedqExpectedHypervolumeImprovementPartitioning(
                model=self.model,
                ref_point=self.ref_point,
                sampler=self.sampler,
                added_x=X[:, 0, :].unsqueeze(1),
                lookahead_sampler=self.lookahead_sampler, )
        else:
            raise ValueError("define approach")

        if self.compute_approach in ["LbJointMCv0", "LbJointMCv1", "LbJointDet", "LbJointDetPartitioned"]:
            average_ehvi_val = average_ehvi(X[:, 1:, :])
        elif self.compute_approach in ["LbNestedMCv0", "LbNestedMCv1", "LbNestedDet"]:
            _, average_ehvi_val = optimize_acqf(
                acq_function=average_ehvi,
                bounds=self.standard_bounds,
                q=self.horizon_length - 1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                sequential=False, )

        total_ehvi = first_step_val + average_ehvi_val
        return total_ehvi