import IMP
import IMP.algebra
import numpy as np


class StandardBD():
    '''
    A class for standard (naive) Brownian dynamics simulations.

    Note: all time units are specified in femtoseconds, unless specied otherwise
          likewise all legth units are specified in angstroms.
    '''

    ### Getters/Setters ###

    def set_dim(self,dim):
        self.dim = dim

    def get_dim(self):
        return self.dim

    
    def set_diffusion(self,D):
        '''
        sets the diffusion coefficient to D in (A^2/fs) units

        :param D: diffusion coefficient. If scalar, applied uniformly to all
          coordinates. If a vector, indicates the diffusion coefficient of each
          coordinate.
        @raise ValueError if the dimensionality of D is not 1 or the dimensionality of the
                          coordinates
        '''
        try:
            if self.get_dim() != len(D):
                raise ValueError(f"Diffusion coefficient for Brownian dynamics must be either scalar"
                             + " or of the same dimensionality as the coordinates")
        except TypeError:
            D = D * np.ones(self.get_dim())
        self.D = np.array(D)

    def get_diffusion(self):
        ''' return diffusion vector for each coordinate '''
        return self.D
    
    def get_core_XYZR(self):
        return self.core_XYZR

    ####################
    ### Constructors ###
    ####################

    def __init__(self,
                 D,
                 kb,
                 dim,
                 core_XYZR,
                 random_steps_taken= [], #TODO: document this
                 temperature= 298.15):
        '''
        BD constructor

        Note: Currently it is assumed all coordinates specify a position in Cartesian space,
              and specified in units of angstroms; this assumption will be generalized in future
              to account for e.g. rotational diffusion, etc.

        :param D: D indicates diffusion coefficient. If scalar, applied uniformly to all
                  coordinates. If a vector, indicates the diffusion coefficient of each
                  coordinate, and its length must be equal to len(initial_coords). 
        :param temperature: simulation temperature in Kelvin
     

        '''
        self.temperature = temperature
        self.set_dim(dim)  # dimension
        self.set_diffusion(D)
        self.random_steps_taken = random_steps_taken
        self.core_XYZR = core_XYZR
        self.kb = kb  # Boltzmann coefficient in TBD units x temperature in Kelvin; TODO: sort the units, here it is assumed kB=1.0 in some units, to work - this should have an effect also on the unit of force
        self.kT = kb*temperature  # Boltzmann coefficient in TBD units x temperature in Kelvin; TODO: sort the units, here it is assumed kB=1.0 in some units, to work - this should have an effect also on the unit of force

    ##############
    # Simulation #
    ##############

    def calc_force_for_standard_BD_step(self,dt,r= None):  # TODO: decide if coords are kept inside the class or not
        """
        TODO: document (Reshef, Dec 2020)
        The banchmark implementation

        :param r: if not None, use this as the random component of the BD step

        :return: force vector, random vector
        """
        gradient = self.core_XYZR.get_derivatives()
        F0 = -1*np.array(gradient)
        dx = F0 * self.D * dt / self.kT
        # if not (type(r) is np.ndarray):
        #     r = self.create_random_vector(dt)
        #
        if r != None:
            r = r
        else:
            r = self.create_random_vector(dt)
        return F0, r, dx

    def create_random_vector(self,dt):
        """
        Return a random vector of length self.dim and normally distributed length with appropriate standard-deviation
        TODO: document (Reshef, Dec 2020)
        """
        # sigma = np.sqrt(2 * self.dim * self.D * dt)  # 2 per dimension due to the Equipartition Theorem
        sigma = np.sqrt(2 * self.D * dt)  # 2 per dimension due to the Equipartition Theorem

        return np.random.normal(0,1,size=(self.dim,)) * sigma
        # return np.array(IMP.algebra.get_random_vector_on_unit_sphere()) * np.sqrt(6* self.D * dt)
        # return np.random.normal(0,1,size=(self.dim,)) * sigma * 0

    def update_particles_locations(self,sbds,locations):
        for i, sbd in enumerate(sbds):
            sbd.get_core_XYZR().set_coordinates(IMP.algebra.Vector3D(locations[i]))

