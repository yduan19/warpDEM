import warp as wp

@wp.func
def periodicBoundaries(x:wp.vec3,x_min:float,x_max:float):
    if x[0]>x_max: 
        x[0] = x[0] - (x_max - x_min)
    elif x<x_min:
        x = x + (x_max - x_min)
    return x

@wp.func
def contact_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    '''
    n: contact normal
    v: relative velocity
    c: penetration depth
    k_n: normal stiffness
    k_d: damping
    k_f: friction stiffness
    k_mu: friction coefficient
    '''
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Tangential displacement force
    ft = k_f * vs
    
    # Coulomb condition - limit friction force
    ft = wp.min(ft, k_mu * wp.abs(fn))

    # total force
    return -n * fn - vt * ft

@wp.kernel
def apply_forces(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_r: wp.array(dtype=float),
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    particle_m: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x = particle_x[i]
    v = particle_v[i]

    f = wp.vec3()

    m1 = particle_m[i]

    # ground contact
    # n = wp.vec3(0.0, 1.0, 0.0)
    # c = wp.dot(n, x)

    # radius=particle_r[i]
    # cohesion_ground = radius
    # cohesion_particle = 0.0

    # if c < cohesion_ground:
    #     f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)

    in_contact = False
    
    # front contact
    n = wp.vec3(1.0, 0.0, 0.0)
    c = wp.dot(n, x)

    radius=particle_r[i]
    cohesion_ground = radius
    cohesion_particle = 0.0

    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)

    # back contact
    n = wp.vec3(1.0, 0.0, 0.0)
    c =3.84- wp.dot(n, x) 

    radius=particle_r[i]
    cohesion_ground = radius
    cohesion_particle = 0.0

    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)

    # left contact
    n = wp.vec3(0.0, 0.0, 1.0)
    c =wp.dot(n, x) 

    radius=particle_r[i]
    cohesion_ground = radius
    cohesion_particle = 0.0
    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)


    # right contact
    n = wp.vec3(0.0, 0.0, -1.0)
    c =20.0-wp.dot(n, x) 

    radius=particle_r[i]
    cohesion_ground = radius
    cohesion_particle = 0.0

    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)


    # # right contact
    # n = wp.vec3(0.0, 0.0, 1.0)
    # c =7.68- wp.dot(n, x) 

    # radius=particle_r[i]
    # cohesion_ground = radius
    # cohesion_particle = 0.0

    # if c < cohesion_ground:
    #     f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)


    # particle contact
    neighbors = wp.hash_grid_query(grid, x, radius * 5.0)

    for index in neighbors:
        if index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_r[index]

            if err <= cohesion_particle:
                n = n / d
                vrel = v - particle_v[index]

                m2 = particle_m[index]
                k_contact_p =k_contact* m1 * m2 / (m1 + m2)
                f = f + contact_force(n, vrel, err, k_contact_p, k_damp, k_friction, k_mu)

    particle_f[i] = f


@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    inv_mass: wp.array(dtype=float),
    r: wp.array(dtype=float),
):
    tid = wp.tid()
    
    ##
  
    n = wp.vec3(0.0, 1.0, 0.0)
    c = wp.dot(n, x[tid])

    radius=r[tid]
    cohesion_ground = radius



    v_new = v[tid] + f[tid] * inv_mass[tid] * dt + gravity * dt
    x_new = x[tid] + v_new * dt

    if c > cohesion_ground:
        v[tid] = v_new
        x[tid] = x_new
    

    
