import warp as wp
import numpy as np

class Particles:
    def __init__(self,x,v,f,r,mass,inv_mass):
        self.x=x
        self.v=v
        self.f=f
        self.r=r
        self.mass=mass
        self.inv_mass=inv_mass

    @staticmethod
    def add_batch( x, v, f, r, mass, inv_mass, x_new, v_new, f_new, r_new, mass_new, inv_mass_new):
        """
        Appends particles from another Particles instance to this instance.

        Parameters:
        particles (Particles): The Particles instance whose data will be added.
        """

        x = x.numpy()
        v = v.numpy()
        f = f.numpy()
        r = r.numpy()
        mass = mass.numpy()
        inv_mass = inv_mass.numpy()

        # Append new data to existing data
        xx = np.concatenate((x_new, x), axis=0)
        vv = np.concatenate((v_new, v), axis=0)
        ff = np.concatenate((f_new, f), axis=0)
        rr = np.concatenate((r_new, r), axis=0)
        massm = np.concatenate((mass_new, mass), axis=0)
        inv_massm = np.concatenate((inv_mass_new, inv_mass), axis=0)

        return wp.array(xx, dtype=wp.vec3), wp.array(vv, dtype=wp.vec3), wp.array(ff, dtype=wp.vec3), wp.array(rr, dtype=float), wp.array(massm, dtype=float), wp.array(inv_massm, dtype=float)

       


    def set_particles_gpu(self):
        x = wp.array(self.x, dtype=wp.vec3)
        v = wp.array(self.v, dtype=wp.vec3)
        f = wp.array(self.f, dtype=wp.vec3)
        r=wp.array(self.r, dtype=float)
        mass=wp.array(self.mass, dtype=float)
        inv_mass=wp.array(self.inv_mass, dtype=float)
        return x,v,f,r,mass,inv_mass

