from StandardBD_IMP import StandardBD
from MSDB_IMP import MSBD
import IMP
import IMP.algebra
import IMP.atom
import IMP.core
import IMP.display
import IMP.rmf
import RMF

import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import numpy as np

T = 298.15 # Kalvin
k = 1.5 * T
# time_in_p_sedc= 450000
time_in_p_sedc = 1800000*10/18*10*2
# time_in_p_sedc = 1





import pickle


def create_p(m):
    p = IMP.Particle(m,"body")
    h0 = IMP.atom.Hierarchy.setup_particle(p)
    IMP.core.XYZR.setup_particle(p).set_radius(1)
    IMP.atom.Mass.setup_particle(p,1)
    IMP.core.XYZ(p).set_coordinates_are_optimized(True)
    rbd = IMP.atom.Diffusion.setup_particle(p)
    return p

def test_bonded(dt,num_of_steps,n):
    stats= np.zeros(num_of_steps)

    """Check brownian dynamics with rigid bodies"""
    # IMP.core.TruncatedHarmonicUpperBound(10., 3., 3., 1.3)
    # IMP.core.SphereDistancePairScore(IMP.core.TruncatedHarmonicUpperBound(10., 3., 3., 1.3))
    # XYZR_a.get_radius()
    m = IMP.Model()
    m.set_log_level(IMP.SILENT)
    pa = create_p(m)
    pb = create_p(m)
    pc = create_p(m)
    pd = create_p(m)
    ps0= IMP.core.HarmonicDistancePairScore(7-3,k*1.8)#7 cause the r of each is 1 plus 5 A 3.8 insted of 1.8
    ps1= IMP.core.SphereDistancePairScore(IMP.core.TruncatedHarmonicUpperBound(10., k,11 ))#10 cause the r of each is 1 plus 8 A 11 thesh fot the upper

    pe = create_p(m)
    pa_1 = create_p(m)
    pb_1 = create_p(m)
    pc_1 = create_p(m)
    pd_1 = create_p(m)
    pe_1 = create_p(m)


    ps2= IMP.core.SphereDistancePairScore(IMP.core.TruncatedHarmonicUpperBound(8., k/16,10+3 ))#10 cause the r of each is 1 plus 8 A 11 thesh fot the upper
    ps3= IMP.core.HarmonicDistancePairScore(12,k/16)#7 cause the r of each is 1 plus 5 A 3.8 insted of 1.8
    ps4= IMP.core.HarmonicDistancePairScore(17,k/64)#7 cause the r of each is 1 plus 5 A 3.8 insted of 1.8


    #bound center k ,treshhold ,limit
    r0 = IMP.core.PairRestraint(m,ps0,(pa,pb))
    r1= IMP.core.PairRestraint(m,ps0,(pb,pc))
    r2 = IMP.core.PairRestraint(m,ps0,(pc,pd))
    r3 = IMP.core.PairRestraint(m,ps1,(pa,pc))
    r4= IMP.core.PairRestraint(m,ps1,(pb,pd))

    r0_ = IMP.core.PairRestraint(m,ps0,(pa_1,pb_1))
    r1_= IMP.core.PairRestraint(m,ps0,(pb_1,pc_1))
    r2_ = IMP.core.PairRestraint(m,ps0,(pc_1,pd_1))
    r3_ = IMP.core.PairRestraint(m,ps1,(pa_1,pc_1))
    r4_= IMP.core.PairRestraint(m,ps1,(pb_1,pd_1))
    #
    r5=IMP.core.PairRestraint(m,ps0,(pd,pe))
    r6=IMP.core.PairRestraint(m,ps1,(pc,pe))

    r5_=IMP.core.PairRestraint(m,ps0,(pd_1,pe_1))
    r6_=IMP.core.PairRestraint(m,ps1,(pc_1,pe_1))

    r7= IMP.core.PairRestraint(m,ps2,(pa,pe))
    r8 = IMP.core.PairRestraint(m,ps2,(pa,pd))
    r9= IMP.core.PairRestraint(m,ps4,(pa,pb))
    r10= IMP.core.PairRestraint(m,ps3,(pa,pb))
    r11 = IMP.core.PairRestraint(m,ps4,(pb,pc))
    r12 = IMP.core.PairRestraint(m,ps3,(pb,pc))


    r7_= IMP.core.PairRestraint(m,ps2,(pa_1,pe_1))
    r8_ = IMP.core.PairRestraint(m,ps2,(pa_1,pd_1))
    r9_= IMP.core.PairRestraint(m,ps4,(pa_1,pb_1))
    r10_= IMP.core.PairRestraint(m,ps3,(pa_1,pb_1))
    r11_ = IMP.core.PairRestraint(m,ps4,(pb_1,pc_1))
    r12_ = IMP.core.PairRestraint(m,ps3,(pb_1,pc_1))


    ps7= IMP.core.HarmonicDistancePairScore(7-3,k*18)#7 cause the r of each is 1 plus 5 A 3.8 insted of 1.8
    r24 = IMP.core.PairRestraint(m,ps7,(pa,pb))
    r25= IMP.core.PairRestraint(m,ps7,(pe,pa_1))
    r26= IMP.core.PairRestraint(m,ps3,(pe,pb_1))
    r27= IMP.core.PairRestraint(m,ps3,(pd,pa_1))
    r28= IMP.core.PairRestraint(m,ps4,(pd,pa_1))





    #
    sf = IMP.core.RestraintsScoringFunction([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,
                                         r24]+[r25,r0_,r1_,r2_,r3_,r4_,r5_,r6_,r7_,r8_,r9_,r10_,r11_,r12_,r26,r27,r28])

    d_a = IMP.atom.Diffusion(pa)
    d_b = IMP.atom.Diffusion(pb)
    d_c = IMP.atom.Diffusion(pc)
    d_d = IMP.atom.Diffusion(pd)
    d_e = IMP.atom.Diffusion(pe)
    d_a_ = IMP.atom.Diffusion(pa_1)
    d_b_ = IMP.atom.Diffusion(pb_1)
    d_c_ = IMP.atom.Diffusion(pc_1)
    d_d_ = IMP.atom.Diffusion(pd_1)
    d_e_ = IMP.atom.Diffusion(pe_1)
    diffusion_a = d_a.get_diffusion_coefficient()
    diffusion_b = d_b.get_diffusion_coefficient()
    diffusion_c = d_c.get_diffusion_coefficient()
    diffusion_d = d_d.get_diffusion_coefficient()
    diffusion_e = d_e.get_diffusion_coefficient()
    diffusion_a_ = d_a_.get_diffusion_coefficient()
    diffusion_b_ = d_b_.get_diffusion_coefficient()
    diffusion_c_ = d_c_.get_diffusion_coefficient()
    diffusion_d_ = d_d_.get_diffusion_coefficient()
    diffusion_e_ = d_e_.get_diffusion_coefficient()
    kt = IMP.atom.get_kt(T)
    XYZR_a = IMP.core.XYZR(pa)
    XYZR_b = IMP.core.XYZR(pb)
    XYZR_c = IMP.core.XYZR(pc)
    XYZR_d = IMP.core.XYZR(pd)
    XYZR_e = IMP.core.XYZR(pe)
    XYZR_a_ = IMP.core.XYZR(pa_1)
    XYZR_b_ = IMP.core.XYZR(pb_1)
    XYZR_c_ = IMP.core.XYZR(pc_1)
    XYZR_d_ = IMP.core.XYZR(pd_1)
    XYZR_e_ = IMP.core.XYZR(pe_1)
    bd_a = StandardBD(kb= kt/T,dim= 3,D= diffusion_a,core_XYZR= XYZR_a)
    bd_b = StandardBD(kb= kt/T,dim= 3,D= diffusion_b,core_XYZR= XYZR_b)
    bd_c = StandardBD(kb= kt/T,dim= 3,D= diffusion_c,core_XYZR= XYZR_c)
    bd_d = StandardBD(kb= kt/T,dim= 3,D= diffusion_d,core_XYZR= XYZR_d)
    bd_e = StandardBD(kb= kt/T,dim= 3,D= diffusion_e,core_XYZR= XYZR_e)
    bd_a_ = StandardBD(kb= kt/T,dim= 3,D= diffusion_a_,core_XYZR= XYZR_a_)
    bd_b_ = StandardBD(kb= kt/T,dim= 3,D= diffusion_b_,core_XYZR= XYZR_b_)
    bd_c_ = StandardBD(kb= kt/T,dim= 3,D= diffusion_c_,core_XYZR= XYZR_c_)
    bd_d_ = StandardBD(kb= kt/T,dim= 3,D= diffusion_d_,core_XYZR= XYZR_d_)
    bd_e_ = StandardBD(kb= kt/T,dim= 3,D= diffusion_e_,core_XYZR= XYZR_e_)
    bds = [bd_a,bd_b,bd_c,bd_d,bd_e]+[bd_a_,bd_b_,bd_c_,bd_d_,bd_e_]



    msbd_a = MSBD(bds,sf,s= 4)



    result= None
    fix = 0
    init= [(1.04825, 5.55673, 5.06083), (-0.108803, 8.17713, 7.9483), (0.565886, 7.71646, 3.56105),
           (-1.63166, 4.41963, 4.35162), (0.718096, 3.81547, 0.990831), (3.0439, 3.77295, -2.3531),
           (5.57112, 4.32292, -5.94871), (5.64131, 2.35056, -2.05306), (7.35712, 1.9863, 1.51503),
           (6.14769, 4.8931, 3.91857)]

    bd_a.update_particles_locations(bds, init)
    for i in range(num_of_steps):
        a = msbd_a.do_one_step_recursively(
                                    x_0= [bds[0].get_core_XYZR().get_coordinates(),bds[1].get_core_XYZR().get_coordinates(),
                                          bds[2].get_core_XYZR().get_coordinates() ,bds[3].get_core_XYZR().get_coordinates(),
                                          bds[4].get_core_XYZR().get_coordinates(),bds[5].get_core_XYZR().get_coordinates(),
                                          bds[6].get_core_XYZR().get_coordinates(),bds[7].get_core_XYZR().get_coordinates(),
                                          bds[8].get_core_XYZR().get_coordinates(),bds[9].get_core_XYZR().get_coordinates()],
                                    dt= dt*(4**n),
                                    n= n,
                                    first_step= result)

        x_s, F_0_s, r_0_s_list, sff, x_2, F_0_2, is_valid = a

        bd_a.update_particles_locations(bds,x_s)

        stats[i] = (IMP.algebra.get_radius_of_gyration(x_s))

        if i%int(10000) == 0 and i >1 :
            # m.update()
            # os.update_always("Init position")
            # Round the values to 3 decimal places
            print(fix / (i + 1))
            print(IMP.algebra.get_radius_of_gyration(x_s))
            print(f'time n={n} ({(i + 1) * 4 ** n * dt * 1e-6:.3f} ns)')

            # print(np.mean(stats_rg),np.max(stats_rg),np.min(stats_rg),np.var(stats_rg))
            mean_value, max_value, min_value, variance = np.mean(stats[:i]), \
                                                         np.max(stats[:i]), np.min(stats[:i]), np.var(stats[:i])

            mean_value = round(mean_value, 3)
            max_value = round(max_value, 3)
            min_value = round(min_value, 3)
            variance = round(variance, 3)

            # Print the rounded values
            print("Mean_rg: {}".format(mean_value), "Max_rg: {}".format(max_value),
                  "Min_rg: {}".format(min_value), "Variance_rg: {}".format(variance))


    return stats,fix/(i+1)
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def autocorr5(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    print(np.where(result <0.2 )[0][0])
    return result

if __name__ == '__main__':

    dts=[0.2]
    # dts=dts_ref
    print(dts)
    for k_ in range(len(dts)):
        dt= dts[k_]
        for i in range(0,5):
            for n in range (3,4):
                # n=5
                print("n=",n,"dt=",dt)
                # print(n, dt)
                your_data_n0 = test_bonded(dt=dt, num_of_steps=int(time_in_p_sedc * (1 / dt) / 4 ** n),
                                           n=n)
                file_name =  "run_num_" + str(i + 1) + "long_10_bolls" + str(
                    your_data_n0[1]) + '_fix)test_4_' + str(dt) + '_mean_with_fix_n=_' + str(n) + '.pickle'
                # with open(file_name, 'wb') as handle:
                #     pickle.dump(your_data_n0, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     print("saved", file_name)

