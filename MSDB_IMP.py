'''
Synopsis: Implementation of multiscale Brownian dynamics (ms-bd) algorithm
Authors: Reshef Mintz, Barak Raveh
Last updated after: Dec 29, 2020
'''

import IMP.atom
import IMP
import RMF
import IMP.rmf
import IMP.core
import IMP.display
import IMP.algebra
from SurrogateForceFunction_IMP import SurrogateForceFunction
import numpy as np
class MSBD():
    '''
    A class for multi scale Brownian dynamics simulations.
    Note: all time units are in  fento seconds  (?), unless specified otherwise
    '''

    ### Getters/Setters ###
    def set_standard_BD(self, sbd):
        self.sbd= sbd

    def get_standard_BD(self):
        return self.sbd

    def set_s(self, s:int):

        # assert(s==4.0)
        self.s = int(s) # TODO: enforce type checking

    def get_s(self):
        return self.s

    # def reset_simulation(self):
    # '''
    # Sets the simulation time to zero, and perform any other tasks
    # needed for resetting the simulation
    # '''
    #     self.simulation_time = 0.0

    ####################
    ### Constructors ###
    ####################

    def __init__(self,
                 sbds,
                 sf,
                 s:int= 4.0,
                 ):
        '''
        MD BD constructor

        :param sbds: standard BD implementation a container and plus for a particle
        :param s: s is the number of steps predicted at each level of the recursion (see algorithm documentation)
        '''
        self.sbds= sbds
        self.sf= sf
        self.set_s(s)
        # self.model=model


    ##############
    # Simulation #
    ##############

    def do_baisc_time_step(self,
                                init_locations,
                                dt,
                                r_0_s_list= None ):
        """
        :param r_0_s_list the random step taken for each particle
        :return: F_0_s - a list of the forces in the step for each particle , r_0_s_list_new, x_0 - the init cords ,x_s
        the cords in the end of the step
        """
        for i, sbd in enumerate(self.sbds):
            sbd.get_core_XYZR().set_coordinates(IMP.algebra.Vector3D(init_locations[i]))

        energy = self.sf.evaluate(True)  # the sf should be of the msbd object
        # self.model.update()
        F_0_s, r_0_s_list_new, x_0, x_s= [], [], [], []

        for j, sbd in enumerate(self.sbds):
            init_p = sbd.get_core_XYZR().get_coordinates()
            x_0.append(init_p)
            # local_gradient= local_XYZR.get_derivatives()
            F0, r, dx = sbd.calc_force_for_standard_BD_step(dt=dt)
            if r_0_s_list != None:
                r = r_0_s_list[j][0]
            r_0_s_list_new.append([r])
            new_location = init_p + dx + r
            sbd.get_core_XYZR().set_coordinates(IMP.algebra.Vector3D(new_location))
            x_s.append(new_location)
            F_0_s.append(F0)
        return F_0_s, r_0_s_list_new, x_0,x_s

    def create_a_small_list(self,big_r_list,indexes):
        if big_r_list != None:
            r_0_1_list= []
            for j in range(len(self.sbds)):
                r_0_1_list.append(big_r_list[j][indexes[0]:indexes[1]])
        else:
            r_0_1_list= None
        return r_0_1_list


    def expand_list(self,r_0_2_list,n,dt):
        r_0_s_list= []
        num_of_steps= self.s**n
        dt_small= dt /num_of_steps
        for j, sbd in enumerate(self.sbds):
            r_0_s_list_j= []
            for i in range(num_of_steps):
                if i < len(r_0_2_list[j]):  # remember the random steps taken so far and take it
                    r= r_0_2_list[j][i]
                else:
                    r= sbd.create_random_vector(dt_small)
                r_0_s_list_j.append(r)  # output list of random vectors
            r_0_s_list.append(r_0_s_list_j)
        return r_0_s_list



    def do_one_step_recursively(self,  # TODO: decide if coords are kept inside the class or not
                                x_0,
                                dt:float, n:int,
                                r_0_s_list= None, first_step= None):
        """
        Perform one step of multiscale BD for time dt, with recursion depth n

        :param x_0: the starting coordinates in the configuration space
        :param dt: the time step in sec at this recursion level (after s time steps of dt/s and depth n-1)
        :param n: the depth of the recursive step (n=0 indicates standard BD)
        :param r_0_s_list: if not None, a list of s^n random vectors of average magnitude appropriate
                       for depth 0 to impose (note: should only be delivered here if its from a correct step)
        :param first_step - if the first step was already made before than it is given to the function
        :return:x_s - the new coordinates after time step dt (s time steps of dt/s), None if an error was detected
                F_0_s - the estimated mean force vector between x_0 and x_s (resolution dt)
                r_0_s_list - a list of all the random steps taken so far (s^n steps appropriate for depth 0),
                             this is equal to the input r_0_s_list if it was not None
                sff - the surrogate force function used to compute this time step (scale corresponding to dt/s),
                      None when n==0
                x_2 - the position after 2 x dt/s time steps,
                      None when n==0
                F_0_2 - the estimated mean force vector between x_0 and x_2 (TODO: resolution 2*dt/s?)
                is_valid a flag that says if the step was checked
        """
        if n == 0:
            is_valid = True
            F_0_s, r_0_s_list_new, x_0,x_s = self.do_baisc_time_step(x_0,dt,r_0_s_list)
            return x_s, F_0_s ,r_0_s_list_new,None, None,None,is_valid

        # Recursively compute two steps of dt/s, and predict a time step of dt in total from it
        # Part I - sample two steps of dt/s:
        # Apply first step:

        if first_step != None: #n-1 the first step was done in the recursive step that called this one
            x_1, F_0_1, r_0_1_list, sff_recursive, x_2_recursive, F_0_2_recursive, is_valid = first_step

        else:
            r_0_1_list = self.create_a_small_list(r_0_s_list, (0, int(self.s ** (n - 1))))
            x_1, F_0_1, r_0_1_list, sff_recursive, x_2_recursive, F_0_2_recursive  , is_valid=  \
                self.do_one_step_recursively(x_0,
                                           dt/self.s,
                                           n-1,
                                           r_0_s_list = r_0_1_list)  # F_0_1 is force from x_0 to x_1 (resolution d)

        result = None

        r_1_2_list = self.create_a_small_list(r_0_s_list,(self.s ** (n - 1), 2 * (self.s ** (n - 1))))
        x_2, F_1_2, r_1_2_list, sff_recursive, x_2_recursive, F_0_2_recursive, is_valid = \
            self.do_one_step_recursively(x_1,
                                                     dt / self.s,
                                                     n - 1,
                                                     r_0_s_list= r_1_2_list,
                                                     first_step= result)  # F_1_2 is force from x_1 to x_2
        r_1_small_2_list = None
        F_0_2 = []
        for j in range(len(self.sbds)):
            F_0_2.append(0.5*(F_0_1[j]+F_1_2[j]))
        sff= SurrogateForceFunction(x_0, x_1, F_0_1, F_1_2) #todo
        if r_0_s_list != None:
            r_0_2_list = r_0_s_list
        else:
            r_0_2_list = []
            for j in range(len(self.sbds)):
                if r_1_small_2_list != None:
                    r_0_2_list.append(r_0_1_list[j] + r_1_2_list[j] + r_1_small_2_list[j])
                else:
                    r_0_2_list.append(r_0_1_list[j] + r_1_2_list[j])

        x_s, F_0_s, r_0_s_list, sff = self.do_step_using_surrogate_force_function(
            x_0= x_0,
            sff= sff,
            dt= dt,
            n= n,
            r_0_2_list= r_0_2_list)
        is_valid = False
        return x_s, F_0_s, r_0_s_list, sff, x_2, F_0_2,is_valid

    def do_step_using_surrogate_force_function(self,
                                                  x_0,
                                                  sff, # surrogate force function
                                                  dt,
                                                  n:int,
                                                  r_0_2_list,
                                                  ):
        """
        doing a step of size dt (duration)
        :param x_0: the starting coordinate in the configuration space
        :param sff: surrogate force function
        :param dt: the total time step in f sec - to be constructed of s^n small steps dt/(s^n)
        :param r_0_2_list: a list of all the s^n/2 random steps taken from x_0 to x_2 (x_2 being the coordinate that follows x_1)
        :param n: the depth of the recursive step
        :param: real_force_func the real force function - for returning the real_forces for debug
        :return:x_s - the new point we reached from the initial_coords after time dt
                F_0_s - the ~mean force between x_0 and x_s
                r_0_s_list - a list of all the random steps taken from x_0 to x_s
                sff - surrogate force fucnction used to evaluate the predicted section of this step
        """
        # assert(len(r_0_2_list)==(self.s**n)/2) #thimk it is a mistake if the step is from fix should be here TODO
        # TODO: currently, all s^n steps are evaluated using sff, possibly, it's enough to integrate s steps and add s random vectors appropriately scaled
        x = x_0  # x is the current coodinate
        num_steps = self.s ** n
        dt_small= (dt/num_steps)
        r_0_s_list, x_s, F_0_s = [], [], []

        for j, sbd in enumerate(self.sbds):
            sum_F_0_s = np.zeros(sbd.get_dim())  # zero vector initialy
            x_curr = x[j]
            for i in range(num_steps):
                F_x = sff.evaluate_only_one_particle(x_curr,j)
                sum_F_0_s = sum_F_0_s + F_x
                if i < len(r_0_2_list[j]):  # remember the random steps taken so far and take it
                    r = r_0_2_list[j][i]
                else:
                    r = sbd.create_random_vector(dt_small)
                if i ==0 :
                    r_0_s_list_j= []
                r_0_s_list_j.append(r)  # output list of random vectors
                # print("F_x", F_x), print("r", r,sbd.get_diffusion(),dt_small)
                x_curr = x_curr + F_x * sbd.get_diffusion() * dt_small / sbd.kT + r
            r_0_s_list.append(r_0_s_list_j)
            x_s.append(x_curr)
            F_0_s.append(sum_F_0_s / num_steps)
        return x_s, F_0_s, r_0_s_list, sff




