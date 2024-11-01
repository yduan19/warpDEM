import warp as wp
import numpy

class RotatingTumbler:
    def __init__(self):
        i = 0
        roughn=0.0015
        gap = 0.015
        nrow= int(2*3.1415926*(0.075/roughn))
        number=nrow*gap/roughn

        points = np.zeros((number, 3))

        for y in range(nrow):
            for x in range(int(gap/roughn)):
                
                
    

    def add_particles(self):
        x = Simulation._particle_grid(dim_x = 10, dim_y = 1, dim_z = 16, lower = (0.0, 0.0, 0.0), radius = self.point_radius)
        v = np.ones([len(x), 3]) * np.array([0.0, 0.0, 0.0])
        f = np.zeros_like(v)

        radius = np.ones(len(x)) * 0.1

        num_particles = len(x)

        # Determine the number of elements to double, e.g., 50% of the total elements
        num_large_particles = int(num_particles * 0.5)

        # Randomly select indices without replacement
        indices_to_double = np.random.choice(len(radius), num_large_particles, replace=False)

        # Double the selected elements
        radius[indices_to_double] = 0.1*2


        # Determine inv_mass
        inv_mass=1.0/(1333*np.pi*radius**3)
        mass = (1333*np.pi*radius**3)


        return x,v,f,radius,mass,inv_mass