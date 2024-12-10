This code is tested using `python 3.9`

Installing the dependencies:

`pip install -r requirements.txt`

Example of a command to run the code:

`python -W ignore main.py --problem RCBD --n_var 3 --n_obj 2 --algo BINOM --look_ahead_horizon 2  --n_iter 110 --n_init 5  --n_seed 1`
`python -W ignore main.py --problem RCBD --n_var 3 --n_obj 2 --algo NMMO --look_ahead_horizon 2  --nmmo_method LbJointDet --n_iter 110 --n_init 5  --n_seed 1`
`python -W ignore main.py --problem RCBD --n_var 3 --n_obj 2 --algo NMMO --look_ahead_horizon 2  --nmmo_method LbNestedDet  --n_iter 110 --n_init 5  --n_seed 1`

The problem name, the number of variable and objectives, algorithm name, algorithm-related parameters, number of iterations, number of initial points, and number of seeds can be changed.
All results are saved in the '\result' folder.

Problem name options: RCBD, FBTD, GTD, WBD, DBD, ZDT3, MOF (number if variabeles and objectives for each problem -except zdt3 variable- can't be changed)

algorithm name options: BINOM, NMMO

For lookahead methods (BINOM, NMMO) add: `--look_ahead_horizon n`

For NMMO methods add: `--nmmo_method ` either LbJointDet or LbNestedDet.
