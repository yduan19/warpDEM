from dataclasses import dataclass
import json
from typing import Tuple

@dataclass
class sys_params:
    frame_dt: float
    frame_count: int
    sim_substeps: int
    sim_dt: float
    sim_time: float
    sim_gravity: Tuple[float, float, float]
    grid: Tuple[float, float, float]
    grid_particle_size_ratio: float



if __name__ == '__main__':
    with open('input/heapflow.json', 'r') as f:
        data=json.load(f)

    data = data['sys_params']
    # data['sim_gravity'] = tuple(data['sim_gravity'])
    # data['grid'] = tuple(data['grid'])

    sys_params = sys_params(**data)

    print(sys_params)