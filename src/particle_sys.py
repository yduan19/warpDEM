from dataclasses import dataclass
import json

@dataclass
class particle_sys:

    point_radius: float
    k_contact: int
    k_damp: int
    k_friction: float
    k_mu: float
    e: float
    

if __name__ == '__main__':
    with open('input/heapflow.json', 'r') as f:
        data=json.load(f)



    particle_param = particle_sys(**data['particle_params'])

    print(particle_param)