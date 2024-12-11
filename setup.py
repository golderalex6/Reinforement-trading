import os
from pathlib import Path
import json

parameters = {
    "symbol":"META",
    "start_date":"2018-01-01",
    "test_date":"2022-01-01",
    "end_date":"2024-01-01",
    "window_size":5
}
with open(os.path.join(Path(__file__).parent,'parameters.json'),'w+') as f:
    json.dump(parameters,f,indent = 4)


if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
    os.mkdir(os.path.join(Path(__file__).parent,'data'))

os.mkdir(os.path.join(Path(__file__).parent,'metadata'))
ppo_metadata = {
        'layers':[100,50,20,10,5],
        'activation':'LeakyReLU',
        'optimizer':'Adam',
        'epochs':32,
        'batch_size':10,
        'learning_rate':0.0003,
        'policy':'MlpPolicy'
    }
with open(os.path.join(Path(__file__).parent,'metadata','ppo_metadata.json'),'w+') as f:
    json.dump(ppo_metadata,f,indent = 4)

a2c_metadata = {
        'layers':[100,50,20,10,5],
        'activation':'LeakyReLU',
        'optimizer':'Adam',
        'learning_rate':0.0003,
        'policy':'MlpPolicy'
    }
with open(os.path.join(Path(__file__).parent,'metadata','a2c_metadata.json'),'w+') as f:
    json.dump(a2c_metadata,f,indent = 4)

dql_metadata = {
        'layers':[100,50,20,10,5],
        'activation':'LeakyReLU',
        'optimizer':'Adam',
        'batch_size':10,
        'learning_rate':0.0003,
        'policy':'MlpPolicy'
    }
with open(os.path.join(Path(__file__).parent,'metadata','dqn_metadata.json'),'w+') as f:
    json.dump(dql_metadata,f,indent = 4)

os.mkdir(os.path.join(Path(__file__).parent,'models'))
