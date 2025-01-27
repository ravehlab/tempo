'''
Synopsis: Implementation of a surrogate force function
Author: Reshef Mintz, Barak Raveh
Last updated after: Dec 21, 2020
'''
import numpy as np
alpha=0.5
# alpha=1 #has to be a param in the class
class SurrogateForceFunction():  # TODO: make this an abstract base class using module abc, then derive from it. eg FirstTaylorSurrogateForceFunction
    def __init__(self, x_0, x_1, F_0_1, F_1_2):
        '''
        Initialize the surrogate force based on force samples  at coords x_0 and x_1,
        :param x_0: the first sampled coordinate
        :param x_1: the second sampled coordinate
        :param F_0_1: the estimated force at x_0 (referred to as F_0 hat in the manuscript)
        :param F_1_2: the estimated force at x_1 (referred to as F_1 hat in the
        '''
        #
        self.x_0 = x_0
        self.x_1 = x_1
        self.F_0_1 = F_0_1
        self.F_1_2 = F_1_2
        self.F_hat_deriv = []
        for i in range(len(x_0)):
            self.F_hat_deriv.append((F_1_2[i] - F_0_1[i]) / (x_1[i] - x_0[i]))
        self.alpha = alpha
        # a numerical estimate of the derivative, hence 'hat'

    def evaluate_only_one_particle(self, x,particle_index):
        """
        predict the surrogate force at coordinate x
        :return: the predicted force vector at x
        """
        force_point_tylor= self.F_0_1[particle_index] + (x - self.x_0[particle_index]) * self.F_hat_deriv[particle_index]
        # F_x = self.F_0_1 + (x - self.x_0) * self.F_hat_deriv  # first-order Taylor's approximation
        # force_point = F_x
        # local_alpha= alpha
        local_alpha=alpha
        # max_force_norm = max(np.linalg.norm(self.F_0_1[particle_index]), np.linalg.norm(self.F_1_2[particle_index]))
        # sum_force_norm = np.linalg.norm(self.F_0_1[particle_index])+np.linalg.norm(self.F_1_2[particle_index])
        # if np.linalg.norm(force_point_tylor) <\
        #         max_force_norm:
        #     local_alpha= 1
            # print(local_alpha)
        half = (self.F_0_1[particle_index] + self.F_1_2[particle_index]) / 2

        if alpha ==1:
            force_point = half
        else:
            norm_tylor = np.linalg.norm(force_point_tylor)
            #the version for the 10 and 5 ballsalpha 0.5

            # force_point= local_alpha * (self.F_0_1[particle_index] + self.F_1_2[particle_index]) / 2 + \
            #             (1-local_alpha) * force_point_tylor/np.linalg.norm(force_point_tylor)


            force_point= local_alpha * half + \
            (1-local_alpha) * force_point_tylor/norm_tylor * \
                         min(np.linalg.norm(half), norm_tylor,np.linalg.norm(np.linalg.norm(self.F_1_2[particle_index])),
                             np.linalg.norm(np.linalg.norm(self.F_0_1[particle_index])))




        return  force_point




        # return force_point*0.9
        # return force_point
        # print( (np.linalg.norm(force_point*2))/\
        #        (np.linalg.norm(self.F_0_1[particle_index])+np.linalg.norm(self.F_1_2[particle_index])))
        # return force_point_tylor * (np.linalg.norm(force_point_tylor))/\
        #        (np.linalg.norm(self.F_0_1[particle_index]+self.F_1_2[particle_index]))



    def evaluate(self, x):
        """
        predict the surrogate force at coordinate x
        :return: the predicted force vector at x
        """
        force_point = []
        for particle_index in range(len(x)):
            # force_point.append(self.F_0_1[particle_index] + (x[particle_index] - self.x_0[particle_index]) * self.F_hat_deriv[particle_index])
            force_point.append(self.evaluate_only_one_particle(x[particle_index],particle_index))
        # F_x = self.F_0_1 + (x - self.x_0) * self.F_hat_deriv  # first-order Taylor's approximation
        # force_point = F_x
        return force_point

    # def evaluate_only_one_particle(self, x,particle_index):
    #     """
    #     predict the surrogate force at coordinate x
    #     :return: the predicted force vector at x
    #     """
    #     force_point = self.F_0_1[particle_index] + (x - self.x_0[particle_index]) * self.F_hat_deriv[particle_index]
    #     # F_x = self.F_0_1 + (x - self.x_0) * self.F_hat_deriv  # first-order Taylor's approximation
    #     # force_point = F_x
    #     return force_point
