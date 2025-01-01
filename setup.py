import os
from pathlib import Path

if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
    os.mkdir(os.path.join(Path(__file__).parent,'data'))

os.mkdir(os.path.join(Path(__file__).parent,'models'))
