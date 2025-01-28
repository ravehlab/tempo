import SurrogateForceFunction_IMP
from StandardBD_IMP import StandardBD
from MSDB_IMP import MSBD
import IMP
import IMP.algebra
import IMP.atom
import IMP.core
import IMP.display
import IMP.rmf
import matplotlib.pyplot as plt
import RMF
from scipy.stats import ks_2samp

import numpy as np
# import plots

T = 298.15 # Kalvin
k = 1.5 * 298.15 
# time_in_p_sedc= 450000
time_in_p_sedc = 1800000*10/18*10
# time_in_p_sedc = 1800000*10/18*3

import pickle


def create_p(m):
    p = IMP.Particle(m,"body")
    h0 = IMP.atom.Hierarchy.setup_particle(p)
    IMP.core.XYZR.setup_particle(p).set_radius(1)
    IMP.atom.Mass.setup_particle(p,1)
    IMP.core.XYZ(p).set_coordinates_are_optimized(True)
    rbd = IMP.atom.Diffusion.setup_particle(p)
    return p

def test_bonded(dt, num_of_steps, n):
    stats = []
    stats_rg = np.zeros(num_of_steps)
    stats_forces = []
    stats_locations = []
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
    ps0 = IMP.core.HarmonicDistancePairScore(7 - 3, k * 1.8)  # 7 cause the r of each is 1 plus 5 A 3.8 instead of 1.8
    ps1 = IMP.core.SphereDistancePairScore(
        IMP.core.TruncatedHarmonicUpperBound(10., k, 11))  # 10 cause the r of each is 1 plus 8 A 11 threshold for the upper

    pe = create_p(m)
    ps2 = IMP.core.SphereDistancePairScore(IMP.core.TruncatedHarmonicUpperBound(center = 8., 
                                                                                k = k / 16,
                                                                                threshold = 10 + 3))  # 10 cause the r of each is 1 plus 8 A 11 threshold for the upper
    ps3 = IMP.core.HarmonicDistancePairScore(12, k / 16)  # 7 cause the r of each is 1 plus 5 A 3.8 instead of 1.8
    ps4 = IMP.core.HarmonicDistancePairScore(17, k / 64)  # 7 cause the r of each is 1 plus 5 A 3.8 instead of 1.8

    r0 = IMP.core.PairRestraint(m, ps0, (pa, pb))
    r1 = IMP.core.PairRestraint(m, ps0, (pb, pc))
    r2 = IMP.core.PairRestraint(m, ps0, (pc, pd))

    #
    r5 = IMP.core.PairRestraint(m, ps0, (pd, pe))

    r6 = IMP.core.PairRestraint(m, ps1, (pc, pe))
    r3 = IMP.core.PairRestraint(m, ps1, (pa, pc))
    r4 = IMP.core.PairRestraint(m, ps1, (pb, pd))
    r7 = IMP.core.PairRestraint(m, ps2, (pa, pe))
    r8 = IMP.core.PairRestraint(m, ps2, (pa, pd))

    r9 = IMP.core.PairRestraint(m, ps4, (pa, pb))
    r10 = IMP.core.PairRestraint(m, ps3, (pa, pb))
    r11 = IMP.core.PairRestraint(m, ps4, (pb, pc))
    r12 = IMP.core.PairRestraint(m, ps3, (pb, pc))

    ps7 = IMP.core.HarmonicDistancePairScore(7 - 3, k * 18)  # 7 cause the r of each is 1 plus 5 A 3.8 instead of 1.8
    r24 = IMP.core.PairRestraint(m, ps7, (pa, pb))

    #
    sf = IMP.core.RestraintsScoringFunction([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12,
                                             r24])

    d_a = IMP.atom.Diffusion(pa)
    d_b = IMP.atom.Diffusion(pb)
    d_c = IMP.atom.Diffusion(pc)
    d_d = IMP.atom.Diffusion(pd)
    d_e = IMP.atom.Diffusion(pe)
    diffusion_a = d_a.get_diffusion_coefficient()
    diffusion_b = d_b.get_diffusion_coefficient()
    diffusion_c = d_c.get_diffusion_coefficient()
    diffusion_d = d_d.get_diffusion_coefficient()
    diffusion_e = d_e.get_diffusion_coefficient()
    kt = IMP.atom.get_kt(T)
    XYZR_a = IMP.core.XYZR(pa)
    XYZR_b = IMP.core.XYZR(pb)
    XYZR_c = IMP.core.XYZR(pc)
    XYZR_d = IMP.core.XYZR(pd)
    XYZR_e = IMP.core.XYZR(pe)
    bd_a = StandardBD(kb=kt / T, dim=3, D=diffusion_a, core_XYZR=XYZR_a)
    bd_b = StandardBD(kb=kt / T, dim=3, D=diffusion_b, core_XYZR=XYZR_b)
    bd_c = StandardBD(kb=kt / T, dim=3, D=diffusion_c, core_XYZR=XYZR_c)
    bd_d = StandardBD(kb=kt / T, dim=3, D=diffusion_d, core_XYZR=XYZR_d)
    bd_e = StandardBD(kb=kt / T, dim=3, D=diffusion_e, core_XYZR=XYZR_e)
    bds = [bd_a, bd_b, bd_c, bd_d, bd_e]
    msbd_a = MSBD(bds, sf, s=4)

    result = None
    fix = 0
    fix_only_angle = 0
    fix_only_norm = 0
    fix_both = 0
    init = [(1.6675, 5.00333, -1.24287), (4.36713, 8.02203, 0.101614), (2.60661, 7.21669, 2.9903),
            (0.816652, 3.79812, 2.35317), (0.0571028, -0.538041, 1.82083)]
    bd_a.update_particles_locations(bds, init)
    for i in range(num_of_steps):
        a = msbd_a.do_one_step_recursively(
            x_0=[bds[0].get_core_XYZR().get_coordinates(), bds[1].get_core_XYZR().get_coordinates(),
                 bds[2].get_core_XYZR().get_coordinates(), bds[3].get_core_XYZR().get_coordinates(),
                 bds[4].get_core_XYZR().get_coordinates()],
            dt=dt * (4 ** n),
            n=n,
            first_step=result)

        x_s, F_0_s, r_0_s_list, sff, x_2, F_0_2, is_valid = a

        bd_a.update_particles_locations(bds, x_s)
        stats_rg[i] = (IMP.algebra.get_radius_of_gyration(x_s))
        if i % int(100) == 0 and i>1:
            print(IMP.algebra.get_radius_of_gyration(x_s))
            print(f'time n={n} ({(i + 1) * 4 ** n * dt * 1e-6:.3f} ns)')

            mean_value, max_value, min_value, variance = np.mean(stats_rg[:i]), \
                                                         np.max(stats_rg[:i]), np.min(stats_rg[:i]), np.var(stats_rg[:i])

            mean_value = round(mean_value, 3)
            max_value = round(max_value, 3)
            min_value = round(min_value, 3)
            variance = round(variance, 3)

            # Print the rounded values
            print("Mean_rg: {}".format(mean_value), "Max_rg: {}".format(max_value),
                  "Min_rg: {}".format(min_value), "Variance_rg: {}".format(variance))
    print("fix_both", fix_both / (num_of_steps))
    print("fix_only_norm", fix_only_norm / (num_of_steps))
    print("fix_only_angle", fix_only_angle / (num_of_steps))
    stats = [stats_rg, stats_locations, stats_forces]

    return stats, fix / num_of_steps

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
    # assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    # print(np.where(result < 0.2 )[0][0])
    result[result > 1]= 1
    result[result < -1]= -1
    return result

def get_taho_half(data,n,step_for_auto_corr,dt):
    unserialized_data_0 = data[0][::step_for_auto_corr]
    estimate_auto_curelation= autocorr5(np.array(unserialized_data_0))
    x_2= np.linspace(0,(len(unserialized_data_0))
                     *step_for_auto_corr*dt*4**n,len(unserialized_data_0))
    # taho= (np.where(estimate_auto_curelation < 0.5 )[0][0]*step_for_auto_corr*dt*4**n)
    if np.isnan(np.array(estimate_auto_curelation)).any():
        taho = 0
    else:
        taho= (np.where(estimate_auto_curelation < 0.5 )[0][0]*step_for_auto_corr*dt*4**n)
        # taho = (np.where(estimate_auto_curelation < 0.5)[0][0] * step_for_auto_corr * dt * 4 ** n)

        plt.plot(x_2,estimate_auto_curelation)
        plt.xlabel('time fsec', fontsize=18)
        plt.xscale("log")
        plt.ylabel('autocorelation', fontsize=16)
        plt.title("n="+str(n)+"di_alanin_half_time"+str(taho)+
                  f'({(len(unserialized_data_0) * 4 ** n * dt*step_for_auto_corr) * 1e-6:.3f} ns)')

        plt.show()

    return x_2,taho,estimate_auto_curelation
def save_file(file_name,data):

    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved",file_name)
if __name__ == '__main__':
    cluster = False

    dts= [0.20]


    for k_ in range(len(dts)):
        dt= dts[k_]

        for i in range(137,138):
                n = 4
                print(n, dt)
                your_data_n5 = test_bonded(dt=dt, num_of_steps=int(time_in_p_sedc * (1 / dt) / 4 ** n),
                                        n=n)
                # x_2_n5, taho_n5, estimate_auto_curelation_n5 = plots.get_taho_half(your_data_n5[0], n, 1, dt)

                file_name= "5_balls_data/" +"run_num_"+str(i+1)+"long_5_bolls"+str(your_data_n5[1])+'_fix)test_6_'+str(dt)+'_mean_with_fix_n=_'+str(n)+'_angle'+'.pickle'
                # with open(file_name, 'wb') as handle:
                #     pickle.dump(your_data_n5, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #     print("saved",file_name)
