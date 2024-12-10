r"""Abstract base module for all BoTorch models.

This module contains `Model`, the abstract base class for all BoTorch models,
and `ModelList`, a container for a list of Models.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
import torch
from botorch import settings
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    InputDataError,
)
from botorch.logging import shape_to_str
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.posteriors import Posterior, PosteriorList
from botorch.sampling.base import MCSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import is_fully_bayesian
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList


if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform  # pragma: no cover

TFantasizeMixin = TypeVar("TFantasizeMixin", bound="FantasizeMixin")


class Model(Module, ABC):
    r"""Abstract base class for BoTorch models.

    The `Model` base class cannot be used directly; it only defines an API for other
    BoTorch models.

    `Model` subclasses `torch.nn.Module`. While a `Module` is most typically
    encountered as a representation of a neural network layer, it can be used more
    generally: see
    `documentation XXXX`_
    on custom NN Modules.

    `Module` provides several pieces of useful functionality: A `Model`'s attributes of
    `Tensor` or `Module` type are automatically registered so they can be moved and/or
    cast with the `to` method, automatically differentiated, and used with CUDA.

    Attributes:
        _has_transformed_inputs: A boolean denoting whether `train_inputs` are currently
            stored as transformed or not.
        _original_train_inputs: A Tensor storing the original train inputs for use in
            `_revert_to_original_inputs`. Note that this is necessary since
            transform / untransform cycle introduces numerical errors which lead
            to upstream errors during training.
        _is_fully_bayesian: Returns `True` if this is a fully Bayesian model.
        _is_ensemble: Returns `True` if this model consists of multiple models
            that are stored in an additional batch dimension. This is true for the fully
            Bayesian models.
    """  # noqa: E501

    _has_transformed_inputs: bool = False
    _original_train_inputs: Optional[Tensor] = None
    _is_fully_bayesian = False
    _is_ensemble = False

    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: For models with an inferred noise level, if True,
                include observation noise. For models with an observed noise level,
                this must be a `model_batch_shape x 1 x m`-dim tensor or
                a `model_batch_shape x n' x m`-dim tensor containing the average
                noise for each batch and output. `noise` must be in the
                outcome-transformed space if an outcome transform is used.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        pass  # pragma: no cover

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define batch_shape property")

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define num_outputs property")

    def subset_output(self, idcs: List[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            A `Model` object of the same type and with the same parameters as
            the current model, subset to the specified output indices.
        """
        raise NotImplementedError

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        raise NotImplementedError(
            f"`condition_on_observations` not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `SupervisedDataset`."""
        from botorch.models.utils.parse_training_data import parse_training_data

        return parse_training_data(cls, training_data, **kwargs)

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Optional[Module] = None,
    ) -> Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    def _revert_to_original_inputs(self) -> None:
        r"""Revert training inputs back to original."""
        if hasattr(self, "input_transform") and self._has_transformed_inputs:
            self.set_train_data(self._original_train_inputs, strict=False)
            self._has_transformed_inputs = False

    def eval(self) -> Model:
        r"""Puts the model in `eval` mode and sets the transformed inputs."""
        self._set_transformed_inputs()
        return super().eval()

    def train(self, mode: bool = True) -> Model:
        r"""Put the model in `train` mode. Reverts to the original inputs if in `train`
        mode (`mode=True`) or sets transformed inputs if in `eval` mode (`mode=False`).

        Args:
            mode: A boolean denoting whether to put in `train` or `eval` mode.
                If `False`, model is put in `eval` mode.
        """
        if mode:
            self._revert_to_original_inputs()
        else:
            self._set_transformed_inputs()
        return super().train(mode=mode)

    @property
    def dtypes_of_buffers(self) -> Set[torch.dtype]:
        return {t.dtype for t in self.buffers() if t is not None}


class UpdatedFantasizeMixin(ABC):
    """
    Mixin to add a `fantasize` method to a `Model`.

    Example:
        class BaseModel:
            def __init__(self, ...):
            def condition_on_observations(self, ...):
            def posterior(self, ...):
            def transform_inputs(self, ...):

        class ModelThatCanFantasize(BaseModel, FantasizeMixin):
            def __init__(self, args):
                super().__init__(args)

        model = ModelThatCanFantasize(...)
        model.fantasize(X)
    """

    @abstractmethod
    def condition_on_observations(
            self: TFantasizeMixin, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> TFantasizeMixin:
        """
        Classes that inherit from `FantasizeMixin` must implement
        a `condition_on_observations` method.
        """

    @abstractmethod
    def posterior(
            self,
            X: Tensor,
            *args,
            observation_noise: bool = False,
            **kwargs: Any,
    ) -> Posterior:
        """
        Classes that inherit from `FantasizeMixin` must implement
        a `posterior` method.
        """

    @abstractmethod
    def transform_inputs(
            self,
            X: Tensor,
            input_transform: Optional[Module] = None,
    ) -> Tensor:
        """
        Classes that inherit from `FantasizeMixin` must implement
        a `transform_inputs` method.
        """

    # When Python 3.11 arrives we can start annotating return types like
    # this as 'Self', but at this point the verbose 'T...' syntax is needed.
    def fantasize(
            self: TFantasizeMixin,
            # TODO: see if any of these can be imported only if TYPE_CHECKING
            X: Tensor,
            sampler: MCSampler,
            observation_noise: Optional[Tensor] = None,
            **kwargs: Any,
    ) -> TFantasizeMixin:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X`, including observation noise.
        If `observation_noise` is a Tensor, use it directly as the observation
        noise to add.
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: A `model_batch_shape x 1 x m`-dim tensor or
                a `model_batch_shape x n' x m`-dim tensor containing the average
                noise for each batch and output, where `m` is the number of outputs.
                `noise` must be in the outcome-transformed space if an outcome
                transform is used.
                If None and using an inferred noise likelihood, the noise will be the
                inferred noise level. If using a fixed noise likelihood, the mean across
                the observation noise in the training data is used as observation noise.
            kwargs: Will be passed to `model.condition_on_observations`

        Returns:
            The constructed fantasy model.
        """
        if not isinstance(observation_noise, Tensor) and observation_noise is not None:
            raise DeprecationError(
                "`fantasize` no longer accepts a boolean for `observation_noise`."
            )
        elif observation_noise is None and isinstance(
                self.likelihood, FixedNoiseGaussianLikelihood
        ):
            if self.num_outputs > 1:
                # make noise ... x n x m
                observation_noise = self.likelihood.noise.transpose(-1, -2)
            else:
                observation_noise = self.likelihood.noise.unsqueeze(-1)
            observation_noise = observation_noise.mean(dim=-2, keepdim=True)
        # if the inputs are empty, expand the inputs
        if X.shape[-2] == 0:
            output_shape = (
                    sampler.sample_shape
                    + X.shape[:-2]
                    + self.batch_shape
                    + torch.Size([0, self.num_outputs])
            )
            Y = torch.empty(output_shape, dtype=X.dtype, device=X.device)
            if observation_noise is not None:
                kwargs["noise"] = observation_noise.expand(Y.shape[1:])
            return self.condition_on_observations(
                X=self.transform_inputs(X),
                Y=Y,
                **kwargs,
            )
        propagate_grads = kwargs.pop("propagate_grads", False)
        with fantasize_flag():
            with settings.propagate_grads(propagate_grads):
            # with torch.no_grad():
                post_X = self.posterior(
                    X,
                    observation_noise=True
                    if observation_noise is None
                    else observation_noise,
                )
            # Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m -> 10 x 1 x 1 x 1
            # print("1, ", Y_fantasized[:3])
            # print("***", Y_fantasized.shape)
            # Y_fantasized = post_X.mean
            mn = post_X.mean
            Y_fantasized = mn.unsqueeze(0).repeat(sampler.sample_shape[0], 1, 1, 1)  # num_fantasies x batch_shape x n' x m
            # print("2, ", Y_fantasized[:3])
            if observation_noise is not None:
                kwargs["noise"] = observation_noise.expand(Y_fantasized.shape[1:])
            res = self.condition_on_observations(X=self.transform_inputs(X), Y=Y_fantasized, **kwargs)
            # print("**", res)
            return res

