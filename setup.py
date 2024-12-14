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

os.mkdir(os.path.join(Path(__file__).parent,'models'))
