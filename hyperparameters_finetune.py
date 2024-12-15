import optuna
from a2c import A2cTrading
import typing

def copy_dict(main:dict) -> dict:
    """
    Recursively creates a deep copy of a nested dictionary.

    Parameters:
    -----------
        main (dict): The input dictionary to be copied. It can contain nested dictionaries.

    Returns:
    --------
        dict: A deep copy of the input dictionary.
    """

    if isinstance(main,dict):
        return {key:copy_dict(value) for key,value in main.items()}
    return main


def parse_params(trial:optuna.trial.Trial,finetune:dict) -> None:

    for key,value in finetune.items():
        if isinstance(value,dict):
            parse_params(trial,finetune[key])
        else:
            if value[0] == 'float':
                finetune[key] = trial.suggest_float(key,value[1],value[2])
            elif value[0] == 'int': 
                finetune[key] = trial.suggest_int(key,value[1],value[2])
            elif value[0] == 'categorical':
                finetune[key] = trial.suggest_categorical(key,value[1])


class HyperFineTune():

    def __init__(self,model:typing.Callable,finetune:dict) -> None:
        '''
        Docstring here
        '''
        self._finetune = finetune
        self._model = model

    def _objective(self,trial:optuna.trial.Trial) -> float:

        config = copy_dict(self._finetune)
        parse_params(trial,config)

        model = A2cTrading(config)
        # print(model._metadata)
        model.learn(total_timesteps = 40000)
        performamce = model.evaluate(show_fig = False)

        return performamce['ROI']

    def fine_tune(self,trials = 70,direction = 'minimize'):
        study = optuna.create_study(direction = direction)
        study.optimize(self._objective,n_trials = trials)
        print(study.best_value)
        print(study.best_params)
        print(study.best_trial)

if __name__ == '__main__':

    finetune = {
            'gamma': ['float',0.8,0.99],
        }

    # hyper = HyperFineTune(A2cTrading,finetune)
    # hyper.fine_tune(trials = 70,direction = 'maximize')

    def objective(trial:optuna.trial.Trial) -> float:

        # config = copy_dict(finetune)
        # parse_params(trial,config)
        # x = round(trial.suggest_float('gamma',0.8,0.99),3)

        finetune = {
                'gamma': 0.99,
            }

        model = A2cTrading(finetune)
        print(model._metadata)

        model.learn(total_timesteps = 10000)
        performamce = model.evaluate(show_fig = False)

        return performamce['ROI']

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective,n_trials = 10)
    print(study.best_value)
    print(study.best_params)
    print(study.best_trial)
