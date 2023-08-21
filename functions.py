import pymodels as _pymodels
import pyaccel as _pyaccel
import matplotlib.pyplot as _plt
import numpy as _np
import scipy.integrate as _scyint
import scipy.special as _special
from mathphys.beam_optics import beam_rigidity as _beam_rigidity


def calc_amp(acc,energy_offsets, hmax, hmin):
    a_def = _np.zeros(energy_offsets.size)
    indices = _np.zeros(energy_offsets.size)
    try:
        for idx,delta in enumerate(energy_offsets):
            twi,*_ = _pyaccel.optics.calc_twiss(accelerator=acc,
                                              energy_offset=delta,
                                              indices='closed')
            if _np.any(_np.isnan(twi[0].betax)):
                raise _pyaccel.optics.OpticsException('error')
            rx = twi.rx
            betax = twi.betax

            a_sup = (hmax - rx)**2 /betax
            a_inf = (hmin - rx)**2 /betax

            a_max = _np.minimum(a_sup,a_inf)
            idx_min = _np.argmin(a_max)
            indices[idx] = idx_min
            a_def[idx] = a_max[idx_min]
    except (_pyaccel.optics.OpticsException, _pyaccel.tracking.TrackingException):
        pass
    return _np.sqrt(a_def), indices

# Função criada em outro programa utilizando tracking

# ---------------------
def track_eletrons(deltas, n_turn, element_idx, model, pos_x=1e-5, pos_y=3e-6):

    #     pos_x and pos_y have default values but it can be varied
    #     here we define the initial conditions for the simulation
    
    orb = _pyaccel.tracking.find_orbit6(model, indices=[0, element_idx])
    orb = orb[:, 1]
    
    rin = _np.zeros((6, deltas.size))
    rin += orb[:, None]
    rin[0] += pos_x
    rin[2] += pos_y
    rin[4] += deltas
    
    track = _pyaccel.tracking.ring_pass(
        model, rin, nr_turns=n_turn, turn_by_turn=True,
        element_offset=element_idx, parallel=True)
    
    par_out, flag, turn_lost, element_lost, plane_lost = track
    
    turnl_element = []

    for i,item in enumerate(turn_lost):
        if item == n_turn and element_lost[i] == element_idx: # ignora elétrons que não foram perdidos
            pass
        else:
            turnl_element.append((item, element_lost[i], deltas[i]))
    
    return turnl_element

# This function will calculate where the electrons are lost using tracking

# The function recives an array containing all the s positions along the ring 

def trackm_elec(acc,deltas, n_turn, lspos):
    results = []
    ind = []
    spos = _pyaccel.lattice.find_spos(acc)
    
    for k, iten in enumerate(lspos):
        
        el_idx = _np.argmin(_np.abs(spos-iten)) # selecting the index to calculate the lost positions
        turnl = track_eletrons(deltas, n_turn, el_idx, acc, pos_x=1e-5, pos_y=3e-6)
        results.append(turnl) # 
        ind.append(el_idx)
        
    return results, ind


# function returns the desired element index of dipoles, quadrupoles, sextupoles and any other 
# element that is desired, the only thing that is needed is to pass a string 

def el_idx_collector(acc, lname):
    all_index = []
    fam_data = _pymodels.si.get_family_data(acc)

    for string in lname:
        element_index = []
        array_idx = _np.array(fam_data[string]['index'], dtype=object)

        for k, lista in enumerate(array_idx):
            length = len(lista)

            if length % 2 != 0:
                ind = int((length-1)/2)
                element_index.append(lista[ind])

            else:
                try:
                    ind = int(length/2 + 1)
                    element_index.append(lista[ind])
                
                except IndexError:
                    ind = int(length/2 - 1)
                    element_index.append(lista[ind])
                    
        all_index.append(element_index)
    
    return all_index


def plot_track(lista_resul, lista_idx, lista_off, param, element_idx, accep):
    # ----------------
    
    acc= _pymodels.si.create_accelerator()
    acc.cavity_on = False
    acc.radiation_on= False
    acc.vchamber_on = False
    twi0,*_ = _pyaccel.optics.calc_twiss(acc,indices='open')
    betax = twi0.betax
    betax = betax*(1/5)
    
    spos = _pyaccel.lattice.find_spos(acc)

    fig, (a1, a2) = _plt.subplots(1, 2, figsize=(10, 5), sharey=True, gridspec_kw={'width_ratios': [1, 3]})
    
    a1.grid(True, alpha=0.5, ls='--', color='k')
    a1.tick_params(axis='both', labelsize=12)
    a2.grid(True, alpha=0.5, ls='--', color='k')
    a2.tick_params(axis='both', labelsize=12)
    a2.xaxis.grid(False)
    a1.xaxis.grid(False)

    a1.set_title(r'$\delta \times$ lost turn', fontsize=16)
    if 'pos' in param:
        for item in lista_resul:
            a1.plot(item[0], item[2]*1e2, 'k.', label = '')
            a1.set_ylabel(r'positive $\delta$ [%]', fontsize=14)
    
    elif 'neg' in param:
        for item in lista_resul:
            a1.plot(item[0], -item[2]*1e2, 'k.', label = '')
            a1.set_ylabel(r'negative $\delta$ [%]', fontsize=14)
            
    a1.set_xlabel(r'n de voltas', fontsize=14)
    

    a2.set_title(r'tracking ', fontsize=16)

    # Tracking graphics
    # ----------------
    
#     defining the acceptance given a point s of the ring

    acp_s = accep[element_idx]
    ind = _np.argmin(_np.abs(lista_off-acp_s))
    a2.plot(spos[lista_idx][:ind], lista_off[:ind]*1e2,'b.', label=r'accep. limit', alpha=0.25)
    
    if 'pos' in param:
        a2.plot(spos[int(lista_resul[1][-1])], lista_resul[2][-1]*1e2, 'r.', label='lost pos. (track)')
        for item in lista_resul:
            a2.plot(spos[int(item[1])], item[2]*1e2, 'r.')
            
    elif 'neg' in param:
        a2.plot(spos[int(lista_resul[1][-1])], -lista_resul[2][-1]*1e2, 'r.', label='lost pos. (track)')
        for item in lista_resul:
            a2.plot(spos[int(item[1])], -item[2]*1e2, 'r.')
            
#     plotting acceptances with _plt.hlines
    
    _plt.hlines(1e2*acp_s, spos[0], spos[-1], color='black', linestyles='dashed', alpha=0.5)
    
    # linear model graphic
    # ----------------     
    
    # plotting beta function 
    a2.plot(spos, _np.sqrt(betax),color='orange', label=r'$ \sqrt{\beta_x}  $')
        
    # plotting magnetic lattice
    _pyaccel.graphics.draw_lattice(acc, offset=-0.5, height=0.5, gca=True)

    # initial position that tracking begins
    
    a2.plot(spos[element_idx], 0, 'ko', label='{}, ({} m)'.format(
        acc[element_idx].fam_name, "%.2f" % spos[element_idx]))
    
    # setting configurations of the graphic
    a2.set_xlabel(r'$s$ [m]', fontsize=14)
    a2.legend(loc='best', ncol=2)

    fig.tight_layout()
    fig.show()
    
def select_idx(list_, param1, param2):
    arr = _np.array(list_)
    
    n_arr = arr[param1:param2+1]
    
    return n_arr

def f_function_arg_mod(kappa, kappam, b1_, b2_):
    tau = (_np.tan(kappa)**2)[:, None]
    taum = (_np.tan(kappam)**2)
    beta = _beam_rigidity(energy=3)[2]
    ratio = tau/taum/(1+tau)
    arg = (2*tau+1)**2 * (ratio - 1)/tau
    arg += tau - _np.sqrt(tau*taum*(1+tau))
    arg -= (2+1/(2*tau))*_np.log(ratio)
    arg *= _np.sqrt(1+tau)
#     arg *= beta* _np.cos(kappa)[:, None]**2
    arg *= 1/(2*_np.sqrt(tau)) * 1/(1+tau)
    arg *= 2* beta* _np.sqrt(tau)

    bessel = _np.exp(-(b1_-b2_)*tau)*_special.i0e(b2_*tau)
    return arg * bessel

def f_integral_simps_l_mod(taum, b1_, b2_):
    kappam = _np.arctan(_np.sqrt(taum))
    _npts = int(9*1000)
    dkappa = (_np.pi/2-kappam)/_npts
    kappa = _np.linspace(kappam, _np.pi/2, _npts+1)
    func = f_function_arg_mod(kappa, kappam, b1_, b2_)

    # Simpson's 3/8 Rule - N must be mod(N, 3) = 0
    val1 = func[0:-1:3, :] + func[3::3, :]
    val2 = func[1::3, :] + func[2::3, :]
    f_int = 3*dkappa/8*_np.sum(val1 + 3*val2, axis=0)

    # # Simpson's 1/3 Rule - N must be mod(N, 2) = 0
    # val1 = func[0::2, :] + func[2::2, :]
    # val2 = func[1::2, :]
    # f_int = dkappa/3*_np.sum(val1+4*val2, axis=0)
    f_int *= 2*_np.sqrt(_np.pi*(b1_**2-b2_**2))*taum
    return f_int

def get_dis(lsps, _npt, accep, norm):
    model = _pymodels.si.create_accelerator()
    model = _pymodels.si.fitted_models.vertical_dispersion_and_coupling(model)
    
    spos = _pyaccel.lattice.find_spos(model, indices='closed')
    _npoints = int((spos[-1]-spos[0])/0.1)
    scalc= _np.linspace(spos[0],spos[-1], _npoints)
    
    daccp = _np.interp(scalc, spos, accep[1])
    daccn = _np.interp(scalc, spos, accep[0])

    beta = _beam_rigidity(energy=3)[2]
    taum_p = (beta*daccp)**2
    taum_n = (beta*daccn)**2

    kappam_p = _np.arctan(_np.sqrt(taum_p))
    kappam_n = _np.arctan(_np.sqrt(taum_n))
    
    ltime = _pyaccel.lifetime.Lifetime(model)
    b1, b2 = ltime.touschek_data['touschek_coeffs']['b1'],ltime.touschek_data['touschek_coeffs']['b2']

    calc_dp, calc_dn = [], []
    deltasp, deltasn = [], []
    indices, indices_model= [], []

    for indx, s in enumerate(lsps):
        
        dif = _np.abs(scalc - s)
        idx = _np.argmin(dif)
        indices.append(idx)
        
        dif1 = _np.abs(spos - s)
        idx_model = _np.argmin(dif1)

#         kappam_p0 = kappam_p[idx]
#         kappam_n0 = kappam_n[idx]
        
        kappam_p0 = 0.00001 # teste sugerido pelo ximenes
        kappam_n0 = 0.00001

        
        if norm:
            kappap = _np.linspace(kappam_p0, _np.pi/2, _npt)
            deltap = 1/beta * _np.tan(kappap)
            kappan = _np.linspace(kappam_n0, _np.pi/2, _npt)
            deltan = 1/beta * _np.tan(kappan)

            y_p = f_function_arg_mod(kappa=kappap,kappam=kappam_p0,b1_=b1[idx],b2_=b2[idx])
            y_n = f_function_arg_mod(kappa=kappan,kappam=kappam_n0,b1_=b1[idx],b2_=b2[idx])
            y_p = y_p.squeeze()
            y_n = y_n.squeeze()

            norm_facp = _scyint.trapz(y_p, deltap)
            norm_facn = _scyint.trapz(y_n, deltan)

    #         Normalizing the probability density function

            y_p /= (norm_facp)
            y_n /= (norm_facn)

            calc_dp.append(y_p)
            calc_dn.append(y_n)
            deltasp.append(deltap)
            deltasn.append(deltan)
            indices_model.append(idx_model)
        
        else:
            kappap = _np.linspace(kappam_p0, _np.pi/2, _npt)
            deltap = 1/beta * _np.tan(kappap)
            kappan = _np.linspace(kappam_n0, _np.pi/2, _npt)
            deltan = 1/beta * _np.tan(kappan)

            y_p = f_function_arg_mod(kappa=kappap,kappam=kappam_p0,b1_=b1[idx],b2_=b2[idx])
            y_n = f_function_arg_mod(kappa=kappan,kappam=kappam_n0,b1_=b1[idx],b2_=b2[idx])
            y_p = y_p.squeeze()
            y_n = y_n.squeeze()
            
            calc_dp.append(y_p)
            calc_dn.append(y_n)
            deltasp.append(deltap)
            deltasn.append(deltan)
            indices_model.append(idx_model)
            
        
    calc_dp = _np.array(calc_dp)
    calc_dn = _np.array(calc_dn)
    deltasp = _np.array(deltasp)
    deltasn = _np.array(deltasn)
    indices = _np.array(indices)
    indices_model = _np.array(indices_model)


    return calc_dp, calc_dn, deltasp, deltasn, indices, indices_model


def plot_hdis(acc, l_index, deltp, f_densp, deltn, f_densn, hp, hn):
    
    fig, axis = _plt.subplots(1,l_index.squeeze().size, figsize=(10,6))
    _plt.suptitle('Comparação entre densidade de probabilidade analítica e de M.C.', fontsize=15)
    spos = _pyaccel.lattice.find_spos(acc, indices='closed')
    nb = int((spos[-1] - spos[0])/0.1)
    scalc = _np.linspace(spos[0],spos[-1], nb)

    for c,iten in enumerate(l_index):
        ax = axis[c]
        
        if c == 0:
            ax.set_ylabel(r'PDF normalizada', fontsize=15)

        idx = _np.argmin(_np.abs(spos[iten] - scalc))

        ax.set_title('{}'.format(acc[iten].fam_name))
        ax.plot(deltp[idx], f_densp[idx], color='blue', label='Analytic ({:.2f} [m])'.format(scalc[idx]))
        
        ax.tick_params(axis='both', labelsize=14)

        ax.hist(hn[idx], density=True, bins=200, color='lightgrey', label='Monte-Carlo')
        ax.plot(-deltn[idx], f_densn[idx], color='blue')
    #     ax.set_yscale('log')

        ax.hist(hp[idx],density=True, bins=200,color='lightgrey')
    #     ax.set_ylim(1e-1,1e3)
        ax.set_xlim(-0.3,0.3)

        ax.set_xlabel(r'$\delta$ [%]', fontsize=15)
        ax.legend()

        ax.grid(axis='y', ls='--', alpha=0.5)
    #     fig.tight_layout()
        _plt.show
    
    return idx

def create_particles(cov_matrix, num_part):
    
    # permute indices to change the order of the columns:
    # [rx, px, ry, py, de, dl]^T -> [px, py, de, rx, ry, dl]^T
    idcs = [1, 3, 4, 0, 2, 5]
    # [px, py, de, rx, ry, dl]^T -> [rx, px, ry, py, de, dl]^T
    idcs_r = [3, 0, 4, 1, 2, 5]
    
    cov_matrix = cov_matrix[:, idcs][idcs, :]
    
    sig_xx = cov_matrix[:3, :3]
    sig_xy = cov_matrix[:3, 3:]
    sig_yx = cov_matrix[3:, :3]
    sig_yy = cov_matrix[3:, 3:]
    inv_yy = _np.linalg.inv(sig_yy)
    
    part1 = _np.random.multivariate_normal(_np.zeros(6), cov_matrix, num_part).T
    # changing the matrices' columns order 
    part2 = part1.copy()

    vec_a = part2[3:]
    new_mean = (sig_xy @ inv_yy) @ vec_a
    new_cov = sig_xx - sig_xy @ inv_yy @ sig_yx

    part2[:3] = _np.random.multivariate_normal(_np.zeros(3), new_cov, num_part).T
    part2[:3] += new_mean
        
    part2 = part2[idcs_r, :]
    part1 = part1[idcs_r, :]

    return part1, part2


def get_cross_section_distribution(psim, _npts=3000):
    beta_bar = 0
    psi = _np.logspace(_np.log10(_np.pi/2 - psim), 0, _npts)
    psi = _np.pi/2 - psi
    psi = psi[::-1] # this step is necessary to reverse the order of psi
    cpsi = _np.cos(psi)
    cross = _np.zeros(cpsi.size)
    if beta_bar > 1e-19: # I think it maybe will be related to the relativistic regime
        cross += (1 + 1/beta_bar**2)**2 * (2*(1 + cpsi**2)/cpsi**3 - 3/cpsi)
    cross += 4/cpsi + 1
    cross *= _np.sin(psi)
    cross = _scyint.cumtrapz(cross, x=psi, initial=0.0)
    cross /= cross[-1]
    return psi, cross


def cross_section_draw_samples(psim, num_part):
    psi, cross = get_cross_section_distribution(psim)
    crs = _np.random.rand(num_part)
    return _np.interp(crs, cross, psi)


def scatter_particles(part1, part2, de_min):
    gamma = 3e9 / 0.510e6
    beta = _np.sqrt(1 - 1/gamma/gamma)
    num_part = part1.shape[1]

    xl1, yl1, de1 = part1[1], part1[3], part1[4]
    xl2, yl2, de2 = part2[1], part2[3], part2[4]

    # calculating the changing base matrix for every particle
    pz1 = _np.sqrt((1+de1)**2 - xl1*xl1 - yl1*yl1)
    pz2 = _np.sqrt((1+de2)**2 - xl2*xl2 - yl2*yl2)
        
    # desired vectors to construct the transformation matrix
    p_1 = _np.vstack([xl1, yl1, pz1])
    p_2 = _np.vstack([xl2, yl2, pz2])
        
    # new coordinate system
    p_j = p_1 + p_2
    p_j /= _np.linalg.norm(p_j, axis=0) # sum
    p_k = _np.cross(p_1.T, p_2.T).T
    p_k /= _np.linalg.norm(p_k, axis=0) # cross product
    p_l = _np.cross(p_j.T, p_k.T).T

    jkl2xyz = _np.zeros([num_part, 3, 3])
    jkl2xyz[:, :, 0] = p_j.T
    jkl2xyz[:, :, 1] = p_k.T
    jkl2xyz[:, :, 2] = p_l.T
    
    # calculating theta and zeta
    theta = xl1 - xl2 
    zeta = yl1 - yl2
    # defining the value of the scattering angle chi
    chi = _np.sqrt(zeta**2 + theta**2)/2

    # draw the scattering angles from uniform distribution:
    phi = _np.random.rand(num_part)* 2*_np.pi
    psi = _np.random.rand(num_part)* _np.pi/2
    
    # draw the psi angle from the cross section probability density:
    # we need to define a maximum angle to normalize the cross section
    # distribution because it diverges when the full interval [0, pi/2]
    # is considered.
    # To estimate this angle we define a threshold for the minimum energy
    # deviation we care about (de_min) and consider the worst case scenario
    # in terms of chi that could create such scattering. 
    # The worst case happens when chi is maximum, because in this way a small
    # scattering angle would create a larger energy deviation.
    # We considered here a chi twice as large as the maximum chi draw from
    # the particles distribution.
    # This method of doing things should be tested and thought about very
    # carefully, though.
    psim = _np.arccos(de_min/gamma/(chi.max()*2))
    fact=psim*2/_np.pi
    print(psim*2/_np.pi)
    psi = cross_section_draw_samples(psim, num_part)
    
    # new momentum in j,k,l (eq. 16 of Piwinski paper)
    gammat = gamma/_np.sqrt(1 + beta*beta*gamma*gamma*chi*chi)
    dp_ = _np.sin(chi[None, :]) * _np.vstack([
        gammat*_np.cos(psi),
        _np.sin(psi)*_np.cos(phi),
        _np.sin(psi)*_np.sin(phi)])
    p_prime1 = _np.zeros((3, chi.size))
    p_prime1[0] = _np.cos(chi)
    p_prime2 = p_prime1.copy()
    p_prime1 += dp_
    p_prime2 -= dp_

    # returning to momentum x,y,z
    pnew_1 = (jkl2xyz @ p_prime1.T[:, :, None]).squeeze().T
    pnew_2 = (jkl2xyz @ p_prime2.T[:, :, None]).squeeze().T

    delta1 = _np.linalg.norm(pnew_1, axis=0) - 1
    delta2 = _np.linalg.norm(pnew_2, axis=0) - 1
        
    part1_new = part1.copy()
    part1_new[1] = pnew_1[0]
    part1_new[3] = pnew_1[1]
    part1_new[4] = delta1
            
    part2_new = part2.copy()
    part2_new[1] = pnew_2[0]
    part2_new[3] = pnew_2[1]
    part2_new[4] = delta2

    return part1_new, part2_new, fact

def histgms(acc,l_spos,num_part, accep, de_min):

    envelopes = _pyaccel.optics.calc_beamenvelope(acc)
    spos=_pyaccel.lattice.find_spos(acc, indices='closed')
    _npoint = int((spos[-1]-spos[0])/0.1)
    scalc = _np.linspace(spos[0], spos[-1], _npoint)
    histsp1=[]
    histsp2=[]
    indices=[]
    for iten in l_spos:
        
        idx_model = _np.argmin(_np.abs(spos - iten))
        idx = _np.argmin(_np.abs(scalc - iten))

        env = envelopes[idx_model]
        part1, part2= create_particles(env, num_part)
        part1_new, part2_new, fact = scatter_particles(part1, part2, de_min)

#         acpp, acpn = accep[1][idx], accep[0][idx]
        
#         check1 = acpp - part1_new[4]
#         check2 = -(acpn - part2_new[4])
#         cond2 = check2<0
#         cond1 = check1<0
        
#         ind1 = _np.where(cond1)[0] # apenas os desvios de energia maiores que a a aceitancia da máquina
#         ind2 = _np.where(cond2)[0]
        
        ind1 = _np.where(part1_new[4]>=0.001) # teste sugerido pelo ximenes
        ind2 = _np.where(part2_new[4]<=-0.001)
        
        histsp1.append((part1_new[4][ind1]))
        histsp2.append((part2_new[4][ind2]))
        indices.append(idx)
        
    hist1=_np.array(histsp1, dtype='object')
    hist2=_np.array(histsp2, dtype='object')
    indices=_np.array(indices)
    
    return histsp1, histsp2, indices