import os
from pathlib import Path
import json

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
        'epochs':32,
        'batch_size':10,
        'learning_rate':0.0003,
        'policy':'MlpPolicy'
    }
with open(os.path.join(Path(__file__).parent,'metadata','a2c_metadata.json'),'w+') as f:
    json.dump(a2c_metadata,f,indent = 4)

dql_metadata = {
        'layers':[100,50,20,10,5],
        'activation':'LeakyReLU',
        'optimizer':'Adam',
        'epochs':32,
        'batch_size':10,
        'learning_rate':0.0003,
    }
with open(os.path.join(Path(__file__).parent,'metadata','dql_metadata.json'),'w+') as f:
    json.dump(dql_metadata,f,indent = 4)

os.mkdir(os.path.join(Path(__file__).parent,'models'))
