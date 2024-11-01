import warp as wp
import warp.render
import numpy as np
import json
from src.params_sys import sys_params
from src.particle_sys import particle_sys
from src.forces import apply_forces, integrate 
from src.particles import Particles
# import keyboard

wp.init()

class Simulation:
    def __init__(self, config='input/heapflow.json'):
        
        with open(config, 'r') as f:
            data = json.load(f)
        
        self.sys_params = sys_params(**data['sys_params'])
        self.particle_sys = particle_sys(**data['particle_params'])

        # Simulation parameters
        self.sim_dt = self.sys_params.sim_dt
        self.sim_substeps = self.sys_params.sim_substeps
        self.sim_end_time = self.sys_params.sim_time
        self.sim_time = 0.0

        self.sim_steps = int(self.sim_end_time / self.sim_dt) 
        self.frame_dt = self.sim_dt * self.sim_substeps
        self.frame_count = int(self.sim_end_time/self.frame_dt)

        # Particle system parameters
        self.point_radius = self.particle_sys.point_radius
        # self.k_contact = self.particle_sys.k_contact
        self.k_damp = self.particle_sys.k_damp
        self.k_friction = self.particle_sys.k_friction
        
        

        tc = 50*self.sim_dt
        self.e = self.particle_sys.e
        self.k_contact = np.square(np.pi/tc)+np.square(np.log(self.e)/tc)
        self.k_friction = 2.0/7.0 * self.k_contact
        self.k_damp = -np.log(self.e)/tc 
        self.k_mu = self.particle_sys.k_mu
        # print(f"stiffness is {self.k_contact}")



        # Grid parameters
        self.grid = wp.HashGrid(128, 128, 128)
        self.grid_cell_size = self.point_radius * 5.0


        # Initialize particle system
        x,v,f,r,mass,inv_mass = self.add_particles()
        self.particles = Particles(x,v,f,r,mass,inv_mass)
        self.x, self.v, self.f, self.r, self.mass, self.inv_mass = self.particles.set_particles_gpu()
        
        #GPU related
        self.use_graph = wp.get_device().is_cuda
        self.use_graph = False
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.gpu_simulate()
            self.graph = capture.graph

        # OPENGL renderer
        self.renderer = wp.render.OpenGLRenderer(
                    camera_pos=(-40.0, 30, 00.0),
                    camera_front=(1, -0.3, 0),
                    draw_axis=False,
                )
        self.renderer.render_ground()

        
    
    def add_particles(self):
        
        x = Simulation._particle_grid(dim_x = 4, dim_y = 1, dim_z = 16, lower = (0.0, 0.0, 0.0), radius = self.point_radius)
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

    @staticmethod
    def _particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter = 0.1):
        points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
        points_t = np.array((points[0], points[1], points[2])).T * radius * 4.8 + np.array(lower)
        points_t = points_t + np.random.rand(*points_t.shape) * radius * jitter

        points_t = points_t.reshape((-1, 3))
        points_t[:,1] = points_t[:,1] + 20
        return points_t

    def gpu_simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                kernel=apply_forces,
                dim=len(self.x),
                inputs=[
                    self.grid.id,
                    self.x,
                    self.v,
                    self.f,
                    self.r,
                    self.k_contact,
                    self.k_damp,
                    self.k_friction,
                    self.k_mu,
                    self.mass
                ],
            )
            wp.launch(
                kernel=integrate,
                dim=len(self.x),
                inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass,self.r],
            )

    def step(self):
        with wp.ScopedTimer("step", active=True):
            with wp.ScopedTimer("grid build", active=False):
                self.grid.build(self.x, self.grid_cell_size)

            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                self.gpu_simulate()

            self.sim_time += self.frame_dt

                


        
            

    
    def run(self):
        t=0
        while self.renderer.is_running:

            t+=1
            if t % 50==0:

                x,v,f,r,mass,inv_mass = self.add_particles()
                # self.particles.add_batch(self.x, self.v, self.f, self.r, self.mass, self.inv_mass)
                self.x, self.v, self.f, self.r, self.mass, self.inv_mass = Particles.add_batch(self.x, self.v, self.f, self.r, self.mass, self.inv_mass, x, v, f, r, mass, inv_mass)

                # x,v,f,r,mass,inv_mass = self.add_particles()
                # self.particles.add_batch(x,v,f,r,mass,inv_mass)
                # # print(self.particles.r.shape)
                # self.x, self.v, self.f, self.r, self.mass, self.inv_mass = self.particles.set_particles_gpu()

            self.step()
            self.render()




            # self.write2bin()
            # if self.sim_time >= self.sim_end_time:
            #     break
        self.renderer.clear()

    def render(self):
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)

        x=self.x.numpy()
        r=self.r.numpy()

        mask=r>0.15
        
        self.renderer.render_points(
            points=x[mask], radius=self.point_radius*2, name="points", colors=(0.8, 0.3, 0.2)
        )
        self.renderer.render_points(
            points=x[~mask], radius=self.point_radius, name="points1", colors=(0.2, 0.2, 0.8)
        )
        self.renderer.end_frame()
        # print(f"number of particles is {len(x)}")
            
    def write2bin(self):
        points=self.x.numpy()
        n=len(points)
        diameter=self.r.numpy()

        # self.render(points,diameter)

        with open("output.bin", "ab") as f:
            f.write(np.array([n], dtype=np.float32).tobytes())
            for i in range(n):
                buffer = np.array([points[i][0], points[i][1], points[i][2], diameter[i]], dtype=np.float32)
                f.write(buffer.tobytes())



if __name__ == '__main__':

    sim=Simulation()
    with wp.ScopedTimer("total", active=True):
        sim.run()
    # sim.add_particles()
    # sim.write2bin()

    # print(sim.sys_params.frame_count)