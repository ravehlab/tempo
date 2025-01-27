from StandardBD_IMP import StandardBD
from MSDB_IMP import MSBD
import argparse
import IMP
import IMP.test
import IMP.npctransport
import IMP.display
import IMP.rmf
import IMP.atom
import RMF
import math
import numpy as np
import IMP.algebra
import pickle





def create_p(m, name, radius=1, mass=1.0, is_opt=True):
    p = IMP.Particle(m, name)
    d = IMP.core.XYZR.setup_particle(p)
    d.set_coordinates_are_optimized(is_opt)
    d.set_radius(radius)
    IMP.atom.Mass.setup_particle(p, mass)
    IMP.atom.Hierarchy.setup_particle(p)
    diff = IMP.atom.Diffusion.setup_particle(p)
    return p


def create_FG(m, k, num_of_particles, radius, distance_between_particles):
    '''
    Create an FG chain with particles restrained by harmonic springs ("beads on a string")
    :param m: IMP model
    :param k: force coefficient in kcal/mol/A^2 for harmonic spring acting on
              consecutive particles
              in kcal/mol/A^2
    :param num_of_particles: number of particles in a chian
    :param radius: radius in A of each particle in the chain
    :param distance_between_particles_in_FG: rest distance between consecutive particle
            centers in A
    :return
        fg_root: a hierarchy root particle
        restraint_list: a list of restraints
    '''
    p_fg_root = IMP.Particle(m, "FG" + str(num_of_particles))
    fg_root = IMP.atom.Hierarchy.setup_particle(p_fg_root)
    for i in range(0, num_of_particles):
        if i == 0:
            p = create_p(m, "fg" + str(i), radius=radius, mass=1, is_opt=False)
        else:
            p = create_p(m, "fg" + str(i), radius, mass=1, is_opt=True)
        fg_root.add_child(p)
    ps = IMP.core.HarmonicDistancePairScore(distance_between_particles, k)
    restraint_list = []
    fgs = fg_root.get_children()
    for i in range(num_of_particles - 1):
        r = IMP.core.PairRestraint(m, ps, (fgs[i], fgs[i + 1]))
        restraint_list.append(r)
    dict = {}
    dict["fg_root"] = fg_root
    dict["fg_restraint_list"] = restraint_list
    return dict


def crate_excluded_vloume_for_all_particles(m, ps, k_excluded, slack=1):
    # create excluded volume for all particles
    evr = IMP.core.ExcludedVolumeRestraint(ps,
                                           k_excluded,
                                           slack, )  # slack affects speed only
    # (slack of close pairs finder)
    return evr
def create_non_specific_rs(m,ps,k,center,thershold):
    """
    Create non-specific restraints for all particles
    :param m:   IMP model
    :param ps:  list of particles
    :param k:    spring constant
    :param center: of the harmonic potential
    :param thershold:
    the force increase linearly from 0 to k between center-thershold and center+thershold goes to zero
    :return: list of restraints
    """
    rs = []
    # create non-specific restraints for all particles
    for i in range(len(ps)):
        radius_a = IMP.core.XYZR(ps[i]).get_radius()
        for j in range(i,len(ps)):
            radius_b = IMP.core.XYZR(ps[j]).get_radius()
            force = IMP.core.SphereDistancePairScore\
                (IMP.core.TruncatedHarmonicUpperBound(center, k, thershold))

            r = IMP.core.PairRestraint(m, force, (ps[i], ps[j]))
            rs.append(r)
    return rs


def create_slab_with_cylynder_pore(m, slab_pore_radius, slab_height,k_slab=1):
    """
    Create slab with cylynder pore
    :param m:
    :param slab_pore_radius:
    :param slab_height:
    :param k_slab:  kcal/mol/A
    :return:
    """
    # create slab with cylynder pore
    p_slab = IMP.Particle(m, "slab")
    slab = IMP.npctransport.SlabWithCylindricalPore.setup_particle \
        (p_slab, slab_height, slab_pore_radius)
    # Create slab restraint
    slabps = IMP.npctransport.SlabWithCylindricalPorePairScore(k_slab)  # ?
    # now i need a restraint of all particels with the slab
    return slab, slabps


def create_rs_for_slab_with_cylinder_pore(m,slabps, slab, ps):
    '''
    :param slabps: Slab pair score
    :param slab: a Slab decorator object to which the slabps is applies
    :param ps: list of particles that interact with slab
    :return: list of restraint between the slab and each particle in ps
    '''
    restarints_list = []
    for p in ps:
        r = IMP.core.PairRestraint(m, slabps,
                                   [slab.get_particle_index(), p.get_index()],
                                   "slab")
        restarints_list.append(r)
    return restarints_list


def create_bounding_box_rs(K_BB, ps, bb):
    # Outer bounding box for simulation:

    # Add enclosing spheres for pbc and outer simulation box
    bb_harmonic = IMP.core.HarmonicUpperBound(0, K_BB)
    outer_bbss = IMP.core.BoundingBox3DSingletonScore(bb_harmonic,
                                                      bb)
    # Restraints - match score with particles:
    rs = []

    rs.append(IMP.container.SingletonsRestraint(outer_bbss,
                                                ps))
    return rs


def point_on_circle(radius, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = radius * math.cos(angle_radians)
    y = radius * math.sin(angle_radians)
    return x, y


def create_diffusing_particles(m, radius_of_diffuse_particles_list_of_list,bb,slab):
    '''
    :param m: IMP model
    :param num_of_diffuse_particles: number of particles to create
    :param radii_of_diffuse_particles_list: list of list radis for the particles
    '''
    ps =[]
    for radius_list in radius_of_diffuse_particles_list_of_list:
        p_radius_root = IMP.Particle(m, "diffusing_particles_Radius" + str(radius_list[0]))
        diffusing_root = IMP.atom.Hierarchy.setup_particle(p_radius_root)
        for i in range(0, len(radius_list)):
            p = create_p(m, "p" + str(i), radius_list[i])
            d = IMP.core.XYZR(p)
            # this will make sure the particle is not in the slab and not in the bb/2
            d.set_coordinates(IMP.algebra.get_random_vector_in(bb))
            while not out_slab(d, slab):
                d.set_coordinates(IMP.algebra.get_random_vector_in(bb))
            diffusing_root.add_child(IMP.atom.Hierarchy.setup_particle(p))
        ps.append(diffusing_root)
    return ps




def create_FGs_in_slab(m, FG_params, slab, distance_between_particles_in_FG):
    '''
    :param m: IMP model
    :param FG_params: dictinory of parameters for the FG chain with keys:
        k, num_of_particles, radius, distance_between_particles
    :param slab: a SlabWithCylindricalPore particle with a cylindrical cavity to which the FG chains will be
                 anchored. The slab represents the nuclear envelope (NE) and the cylindrical
                 cavity represents the nuclear pore complex (NPC).
    :param distance_between_particles_in_FG:
    :return
        fg_roots: list of roots of FG chains
        fg_restraints: list of harmonic restraints on all pairs of consecutive FG particles
                       in the various chains

    '''
    k = FG_params["k"]
    num_of_particles = FG_params["num_of_particles"]
    radius = FG_params["radius"]
    distance_between_particles = FG_params["distance_between_particles"]
    slab_height = slab.get_thickness()
    # init the FG location to be in the slab
    # z_upper = slab_height / 2.0 + radius - .1
    z_upper = 0
    # z_lower = -slab_height/2.0 - radius + .1
    inner_ring = slab.get_pore_radius()

    # move the FG to the slab
    # create 8 FGs
    fg_roots = []
    fg_restraints = []
    for i in range(0, 8):
        fg_dict = create_FG(m, k, num_of_particles, radius, distance_between_particles)
        cur_root = fg_dict["fg_root"]
        cur_restraints = fg_dict["fg_restraint_list"]
        fg_roots.append(cur_root)
        fg_restraints.extend(cur_restraints)
    # the first ball in the entrence of the pore
    # the second ball in the entrence of the pore
    # put all 8 FGs in the slab in the right location symetricly around the pore entrence z_upper of z axis
    for i, fg_root in enumerate(fg_roots):
        angle_degrees = 45 * i
        x, y = point_on_circle(inner_ring, angle_degrees=angle_degrees)
        for j, p in enumerate(fg_root.get_children()):
            if x == 0:
                IMP.core.XYZ(p).set_coordinates(
                    IMP.algebra.Vector3D(x, y + j * distance_between_particles_in_FG, z_upper))
            elif y == 0:
                IMP.core.XYZ(p).set_coordinates(
                    IMP.algebra.Vector3D(x + j * distance_between_particles_in_FG, y, z_upper))
            else:
                IMP.core.XYZ(p).set_coordinates(IMP.algebra.Vector3D(x + distance_between_particles_in_FG / 2 * j
                                                                     , y + distance_between_particles_in_FG / 2 * j,
                                                                     z_upper))
    dict = {}
    dict["fgs_roots"] = fg_roots
    dict["fgs_restraints"] = fg_restraints
    return dict


def out_slab(p, slab):
    ''' verify particle p is out of slab.
        :param p: particle that is assumed to be decorated by IMP.core.XYZR
        :param slab: a slab with cylindrical pore particle
        :param radius:
        '''
    d = IMP.core.XYZR(p)
    slab_height = slab.get_thickness() *1.12
    pore_radius = slab.get_pore_radius()
    radius = d.get_radius()
    c = d.get_coordinates()
    if c[2] > slab_height / 2.0 + radius - .1:
        return True
    if c[2] < -slab_height / 2.0 - radius + .1:
        return True
    rxy = (c[0] ** 2 + c[1] ** 2) ** .5
    if rxy + radius < pore_radius + .1:
        return True
    # print("out_slab is False - c: ", c, " rxy,max_rxy: ", rxy,
    #       slab.get_pore_radius()-radius, " z,min_z:", c[2], slab_height/2.0+radius)
    return False

def check_transport_acuurance(ps,slab,previous_is_above_the_slab):
    #for each particle check on witch side of the slab it is
    is_above_the_slab = []
    slab_height = slab.get_thickness()
    for j,p in enumerate(ps):
        d = IMP.core.XYZR(p)
        radius = d.get_radius()
        p_location_z = d.get_coordinates()[2]
        if not out_slab(p, slab): #if is in slab
            is_above_the_slab.append(previous_is_above_the_slab[j])
        elif p_location_z >0:
            is_above_the_slab.append(1)
        else:
            is_above_the_slab.append(-1)
    return np.array(is_above_the_slab)
def check_for_transport(is_above_the_slab_1,is_above_the_slab_2,radius_of_disuse_particles_list):
    #make that both arrays are np arrays
    is_above_the_slab_1 = np.array(is_above_the_slab_1)
    is_above_the_slab_2 = np.array(is_above_the_slab_2)

    #chcek if there is a transport 1 change to zero or zero to 1
    was_transport = is_above_the_slab_1 + is_above_the_slab_2
    #get all indexes that are zero and were not zero in is_above_the_slab_1
    indexes_of_transport_ = np.where(was_transport == 0)[0]
    #check if there is a change in the indexes of transport
    indexes_ib_npc = np.where(is_above_the_slab_1 == 0)[0]
    #get the index that are in indexes_of_transport but no in indexes_ib_npc
    indexes_of_transport = np.setdiff1d(indexes_of_transport_,indexes_ib_npc)
    #convert list of lists to 1 d np array
    radius_of_disuse_particles_np_1_d = np.hstack(radius_of_disuse_particles_list)
    #get the radius of the particles that were transported and build a dict taht will be returned with thr adius value as ket and number
    #of transported particles as value
    if indexes_of_transport.size == 0:
        return {}
    radius_of_disuse_particles = radius_of_disuse_particles_np_1_d[indexes_of_transport]

    dict = {}
    for r in radius_of_disuse_particles:
        if r in dict:
            dict[r] += 1
        else:
            dict[r] = 1

    return dict


def create_npc_model(radius_of_disuse_particles_list,
                     L,
                     slab_params,
                     k_excluded,
                     FG_params,
                     distance_between_particles_in_FG,
                    K_BB,
                     strengh_non_specific_interaction=0.002,
                    center_non_specific_interaction = 3,
                    thershold_non_specific_interaction = 5,
                    num_of_fg = 8,
                    rmf_filename= "test.rmf3",

):
    """

    :param radius_of_disuse_particles_list:
    :param L: cube size
    :param slab_params:
    :param k_excluded:
    :param FG_params:
    :param distance_between_particles_in_FG:
    :param K_BB:
    :param strengh_non_specific_interaction:
    :param center_non_specific_interaction:
    :param thershold_non_specific_interaction:
    :param num_of_fg:
    :param rmf_filename:
    :return:
    """

    # Model params:
    num_of_diffuse_particles = len(radius_of_disuse_particles_list)
    type_of_diffuse_particles = np.unique(radius_of_disuse_particles_list)


    # RMF config:
    rmf_fname = rmf_filename
    rmf = RMF.create_rmf_file(rmf_fname)
    # Create model
    m = IMP.Model()
    # Create bb:
    bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-L / 2, -L / 2, -L / 2),
                                   IMP.algebra.Vector3D(L / 2, L / 2, L / 2))
    bb_half = IMP.algebra.BoundingBox3D(-1 * IMP.algebra.Vector3D(L / 2, L / 2, -0.1*L / 2),
                                        IMP.algebra.Vector3D(L / 2, L / 2, L / 2))
    # Create slab:
    slab, slabps = create_slab_with_cylynder_pore(m=m,**slab_params)
    # create particles with different masses and radiuses and make sure they are not in the slab and not in the bb

    # Create particles:
    ps_diffuse_roots = create_diffusing_particles(m,
                                            radius_of_disuse_particles_list,
                                            bb_half,slab)
    fgs_dict = create_FGs_in_slab \
        (m,
         FG_params=FG_params,
         slab=slab,
         distance_between_particles_in_FG=distance_between_particles_in_FG)
    fg_roots = fgs_dict["fgs_roots"]
    fg_restraints = fgs_dict["fgs_restraints"]
    ps_fgs = [x.get_particle() for fg_root in fg_roots for x in fg_root.get_children()]
    ps_diffuse = [x.get_particle() for ps_diffuse_root in ps_diffuse_roots for x in ps_diffuse_root.get_children()]
    # Concat children of all FGs with other diffusers:
    ps_all = ps_diffuse + ps_fgs
    # Add restraints:
    restarints_list_slab = create_rs_for_slab_with_cylinder_pore(m,slabps, slab, ps_all)
    rs_excluded_v = crate_excluded_vloume_for_all_particles(m, ps_all, k_excluded, slack=1)
    rs_bounding_box = create_bounding_box_rs(K_BB, ps_all, bb)
    rs_non_specific_interactions = create_non_specific_rs(m,ps=ps_all,k=strengh_non_specific_interaction,
                                                          center=center_non_specific_interaction,
                                                          thershold=thershold_non_specific_interaction)
    # rs_non_specific_interactions = []
    # TODO add non-specific interactions
    all_rs_list = restarints_list_slab + [rs_excluded_v] + rs_bounding_box + fg_restraints + rs_non_specific_interactions
    sf = IMP.core.RestraintsScoringFunction(all_rs_list)

    IMP.rmf.add_hierarchies(rmf, fg_roots)
    IMP.rmf.add_hierarchies(rmf, ps_diffuse_roots)
    IMP.rmf.add_geometry(rmf, IMP.display.BoundingBoxGeometry(bb))
    IMP.rmf.add_geometry(rmf, IMP.npctransport.SlabWithCylindricalPoreWireGeometry(height=slab.get_thickness(),
                                                                                   radius=slab.get_pore_radius(),
                                                                                   length=L))
    # for p in ps_fgs:
    #     IMP.display.Colored(p).set_color(IMP.display.Color(0,255,0))
    os = IMP.rmf.SaveOptimizerState(m, rmf)
    os.update_always("Init position")
    os.set_log_level(IMP.SILENT)
    # Save rmf file:
    m.update()
    os.update_always("Init position")
    # Optimize:
    print("optimizing")
    # Create Simulation:
    bd = IMP.atom.BrownianDynamics(m)
    bd.set_maximum_time_step(2000.0) #20 looks good
    bd.set_temperature(297.25)
    bd.set_scoring_function(sf)
    bd.optimize(5000000.0)
    # bd.optimize(300000)
    # bd.optimize(30000)
    dict_to_return = {"model": m,"scoring_function":sf, "bounding_box":bb,
                        "fg_roots":fg_roots,
                        "ps_diffuse_roots":ps_diffuse_roots,
                        "rmf_filename":rmf_fname,"optimizer_state":os,
                        "ps_all":ps_all,
                      "slab":slab}
    return dict_to_return

def run_npc(num_frames=1, T=298.15, rmf_filename="test.rmf3",n=0,dt=2000,pikle_file_name="test"):
    kt = IMP.atom.get_kt(T)
    radius_of_disuse_particles_list = [[10]*10,[12]*10,[16]*10,[20]*10]
    # radius_of_disuse_particles_list = [[14]*20,[18]*20]
    radius_of_disuse_particles_list = [[8]*5,[10]*5,[12]*5,[14]*5,[16]*5,[18]*5,[20]*5]
    # Slab params:
    L = 200 *1.3 # for the bounding box
    #slab params dict:
    slab_params = {"slab_pore_radius": 90,
                     "slab_height": 55,
                        "k_slab": 5}

    k_excluded = 1
    # FG params:
    FG_params = {"k": 2,
                 "num_of_particles": 10,
                 "radius":  8,
                 "distance_between_particles": 16*1.9}
    #  (1, 10, 1, 1)
    distance_between_particles_in_FG = 1
    K_BB = 2

    #non specific interactions params:
    strengh_non_specific_interaction = 0.002
    center_non_specific_interaction = 3
    thershold_non_specific_interaction = 5
    num_of_fg = 8 #todo chage to be a param in the code


    # Simulation params:
    angle = (0.5 * 10 ** -10,0.5 * 10 ** -10)
    # angle = (0.97,0.97)
    # norm = (5 * 10 ** 10,5 * 10 ** 10)
    norm = (5 * 10 ** 10,5 * 10 ** 10)
    # n=0
    # dt = 2000
    # num_frames = int(895 *6000/dt)
    # num_frames = int(895)



    dict_pram_for_npc_model= {"radius_of_disuse_particles_list":radius_of_disuse_particles_list,
                                "L":L,
                                "slab_params":slab_params,
                                "k_excluded":k_excluded,
                                "FG_params":FG_params,
                                "distance_between_particles_in_FG":distance_between_particles_in_FG,
                                "K_BB":K_BB,
                                "strengh_non_specific_interaction":strengh_non_specific_interaction,
                                "center_non_specific_interaction":center_non_specific_interaction,
                                "thershold_non_specific_interaction":thershold_non_specific_interaction,
                                "num_of_fg":num_of_fg,
                                "rmf_filename":rmf_filename}
    model_dict = create_npc_model(**dict_pram_for_npc_model)
    #unpack the model dict:
    m = model_dict["model"]
    sf = model_dict["scoring_function"]
    bb = model_dict["bounding_box"]
    fg_roots = model_dict["fg_roots"]
    ps_diffuse_roots = model_dict["ps_diffuse_roots"]
    rmf_fname = model_dict["rmf_filename"]
    os = model_dict["optimizer_state"]
    ps_all = model_dict["ps_all"]
    slab = model_dict["slab"]

    ps_fgs = [x.get_particle() for fg_root in fg_roots for x in fg_root.get_children()]
    ps_diffuse = [x.get_particle() for ps_diffuse_root in ps_diffuse_roots for x in ps_diffuse_root.get_children()]

    bds = []


    for atom in ps_all:
        # assert (IMP.atom.Atom.get_is_setup(atom))
        # D =IMP.atom.Diffusion(atom).get_diffusion_coefficient()

        d = IMP.core.XYZR(atom)
        if d.get_coordinates_are_optimized():
            bds.append(StandardBD(kb=kt / T, dim=3, D=IMP.atom.Diffusion(atom).get_diffusion_coefficient()
                              , core_XYZR=IMP.core.XYZR(atom)))



    result = None
    msbd_a = MSBD(bds, sf, s=4)

    init_difusion = [1]* len(ps_diffuse)

    is_above_the_slab_1 = check_transport_acuurance(ps=ps_diffuse,
                                                    slab=slab,
                                                    previous_is_above_the_slab=init_difusion)
    dicts_of_transport = []
    fg_rg =np.zeros((num_frames,num_of_fg))
    num_of_diffuse_particles = len(ps_diffuse)
    location_of_all_diffuse_particles = np.zeros((num_frames,num_of_diffuse_particles,3))


    len_of_optimizing_cycle = 768
    len_of_optimizing_cycle = 64*3
    for i in range(0, num_frames):
        p = ps_diffuse[1]
        d = IMP.core.XYZR(p)
        fix = 0
        for y in range(0, int(math.ceil((2000/dt)*len_of_optimizing_cycle/(4**n)))):
            a = msbd_a.do_one_step_recursively(
                x_0=[x.get_core_XYZR().get_coordinates() for x in bds],
                dt=dt * (4 ** n),
                n=n,
                first_step=result)

            x_s, F_0_s, r_0_s_list, sff, x_2, F_0_2, is_valid = a
            #update the coordinates of the particles:
            bds[0].update_particles_locations(bds,x_s)



        print(np.mean(fg_rg[:i,:]))
        # Save rmf file:
        m.update()
        os.update_always("Init position")
        # Save stats: 1)rg of fgs 2) transport of particles 3) location of particles
        #rg of fgs
        for j in range(0,len(fg_roots)):
            # loc_of_rgs=IMP.core.XYZ(fg_roots[j]).get_coordinates()
            loc_of_rgs = [IMP.core.XYZ(x).get_coordinates() for x in fg_roots[j].get_children()]
            fg_rg[i,j] = IMP.algebra.get_radius_of_gyration(loc_of_rgs)

        #transport of particles
        is_above_the_slab_2 = check_transport_acuurance(ps=ps_diffuse,
                                                    slab=slab,
                                                    previous_is_above_the_slab=is_above_the_slab_1)
        dict =\
            check_for_transport(is_above_the_slab_1, is_above_the_slab_2, radius_of_disuse_particles_list)
        dicts_of_transport.append(dict)

        #location of particles
        for j in range(0,len(ps_diffuse)):
            location_of_all_diffuse_particles[i,j,:] = IMP.core.XYZ(ps_diffuse[j]).get_coordinates()

        is_above_the_slab_1 = is_above_the_slab_2

        print(dict)

    combined_dict = {}
    # Merge the dictionaries
    for d in dicts_of_transport:
        for key, value in d.items():
            if key in combined_dict:
                # Handle the conflict, for example, you can sum the values
                combined_dict[key] += value
            else:
                combined_dict[key] = value
    print(combined_dict)
    if n == 0:
        alpha = 1
    elif sff == None:
        alpha = None
    else:
        alpha = sff.alpha
    # Save the results:
    stat_to_save = {"fg_rg":fg_rg,
                    "location_of_all_diffuse_particles":location_of_all_diffuse_particles,
                    "combined_dict":combined_dict,
                    "n":n,
                    "dt":dt,
                    "num_frames":num_frames,
                    "param for the model":dict_pram_for_npc_model,
                    "angle":angle,
                    "norm":norm,
                    "dicts_of_transport":dicts_of_transport,
                    "radius_of_disuse_particles_list":radius_of_disuse_particles_list,
                    "alpha":alpha,
                    "total_sim_time":int(math.ceil((2000/dt)*len_of_optimizing_cycle/(4**n)))*dt * (4 ** n)*num_frames}
    #pikle_file_name
    pikle_file_name_to_save = pikle_file_name+"_"+str(n)+"_"+str(dt)+"_.pkl"
    with open(pikle_file_name_to_save, 'wb') as f:
        pickle.dump(stat_to_save, f)
    print("done",pikle_file_name_to_save)
    assert (x_s[0][0] == bds[0].get_core_XYZR().get_coordinates()[0])
    return stat_to_save


if __name__ == '__main__':
    #get num_frames=, T, rmf_filename,n,dt,pikle_file_name=
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", help="depth of the recursive step", nargs='?', type=int, const=1)
    parser.add_argument("--time_step", help="the basic time step size", nargs='?', type=float, const=1)
    parser.add_argument("--temperature", help="depth of the recursive step", nargs='?', type=float, const=1)
    parser.add_argument("--pikle_file_name", help="the name of the file to write to", nargs='?', type=str, const=1)
    parser.add_argument("--rmf_filename", help="the name of the file to write to", nargs='?', type=str, const=1)
    parser.add_argument("--num_frames", help="the length of the simulation", nargs='?', type=int, const=1)
    args = parser.parse_args()
    n = args.depth
    dt = args.time_step
    T = args.temperature
    pikle_file_name = args.pikle_file_name
    rmf_filename = args.rmf_filename
    num_frames = args.num_frames

    # #get num_frames=, T, rmf_filename,n,dt,pikle_file_name=
    data=run_npc(n=n,dt=dt,T=T,pikle_file_name=pikle_file_name,rmf_filename=rmf_filename,num_frames=num_frames)
    #an example of the params:
    #--depth 0 --time_step 2000 --temperature 298.15 --pikle_file_name "test" --rmf_filename "test.rmf" --num_frames 895











