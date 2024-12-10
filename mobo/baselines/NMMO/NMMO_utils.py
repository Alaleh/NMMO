from botorch.utils.transforms import normalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from mobo.baselines.NMMO.UpdatedSingleTaskGP import UpdatedSingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def initialize_model(train_x, train_obj, problem, det=False):
    """
        det: whether we are running the deterministic version of the algorithm or not
    """
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        if det:
            models.append(UpdatedSingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1)))
        else:
            models.append(SingleTaskGP(train_x, train_y,outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model
