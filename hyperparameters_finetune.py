import optuna

# class HyperFineTune():
#     def __init__(self) -> None:
#         '''
#         Docstring here
#         '''
#         pass

def objective(trial):
    x = trial.suggest_float('x',-10,10)
    return (x-2)**2

study = optuna.create_study()
study.optimize(objective,n_trials = 100)
print(study.best_value)
print(study.best_params)
print(study.best_trial)
print(study.best_trials)
