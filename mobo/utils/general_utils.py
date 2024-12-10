from botorch.utils.sampling import draw_sobol_samples
from botorch.test_functions.multi_objective import ZDT3
from mobo.utils.benchmarks import FBTD, GTD, RCBD, WBD, DBD, MOF


def generate_initial_data(problem, n=0):
    if n < 1:
        n = 2 * problem.dim + 1
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    return train_x, train_obj_true


def get_test_problem(num, d=0, M=0, maximize=True):
    test_sets = {"ZDT3": ZDT3(dim=d, negate=maximize),
                 "RCBD": RCBD(negate=maximize),
                 "FBTD": FBTD(negate=maximize),
                 "WBD": WBD(negate=maximize),
                 "DBD": DBD(negate=maximize),
                 "GTD": GTD(negate=maximize),
                 "MOF": MOF(negate=maximize)}

    if num not in test_sets:
        raise "Undefined problem"

    return test_sets[num]


def write_to_file(data, file_path):
    with open(file_path, "a") as filehandle:
        for v in data:
            filehandle.write(','.join([str(i.item()) for i in v]))
            filehandle.write('\n')
    filehandle.close()
