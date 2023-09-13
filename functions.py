import pymodels as _pymodels
import pyaccel as _pyaccel
import matplotlib.pyplot as _plt
import numpy as _np
import scipy.integrate as _scyint
import scipy.special as _special
from mathphys.beam_optics import beam_rigidity as _beam_rigidity
from mathphys.functions import save_pickle, load_pickle
import pandas as pd


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
    spos = _pyaccel.lattice.find_spos(acc, indices='open')
    
    for k, iten in enumerate(lspos):
        
        el_idx = _np.argmin(_np.abs(spos-iten)) # selecting the index to shift the tracking simulation
        turnl = track_eletrons(deltas, n_turn, el_idx, acc)
        results.append(turnl) # 
        
    return results

# this function will sellect the index by the array containing the names of the elements along the ring that people desire to study
# please let's try to run away from the repetitive structures that

# certo eu vou ter todos os indices referentes ao elementos que eu preciso e quero analisar, mas qual eu devo escolher,
# devo tentar selecionar o trecho em que o beta é mínimo para obter 

# sabendo que a função beta é a envoltória do feixe e, portanto, os pontos onde o feixe se encontra em posições de menor beta são as posições onde podem
# ocorrer com maior probabilidade os espalhamentos touschek, lembrando que a envoltória é quem rege o quanto de espaço estes elétrons possuem para oscilar
# Dessa forma, menor a envoltória menor também será o espaço que os elétrons podem realizar suas oscilações betatron aumentando a densidade do feixe em regiões 
# de baixo beta

# so the lname passed to the function must be an array 

# lname deve ser a lista de elementos que será passada para a função ['BC','B1','B2'] por exemplo
    

# function returns the desired element index of dipoles, quadrupoles, sextupoles and any other 
# element that is desired, the only thing that is needed is to pass a string 

# eu preciso pensar se vale a pena colocar todos os indices que eu estou pensando em colocar ou se o pymodels da conta do recado
# eu tenho que me perguntar se é interessante obter os indices de elementos como lkkp por exemplo

def el_idx_collector(acc, lname):
    all_index = []

    if 'mia' in lname:
        all_index.append(_pyaccel.optics.find_indices(acc, 'fam_name', 'mia'))
    elif'mib' in lname:
        all_index.append(_pyaccel.optics.find_indices(acc, 'fam_name', 'mib'))
    elif 'mip' in lname:
        all_index.append(_pyaccel.optics.find_indices(acc, 'fam_name', 'mip'))

    fam_data = _pymodels.si.get_family_data(acc)

    for string in list(lname):
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

# this function recieves a list and search all elements checking if there are strings into the list
   
def char_check(elmnt):
    for char in elmnt:
        returnval = type(char)
        if returnval is str:
            return str
        elif returnval is float or returnval is int:
            return float

# this function will plot the tracking simultation with the linear model calculated plus the touschek scattering distribution
# in the first graphic the plot will be the touschek scattering distribution 

def plot_track(acc, lista_resul, lista_idx, lista_off, param, element_idx, accep, delt, f_dens):
    # ----------------
    
    twi0,*_ = _pyaccel.optics.calc_twiss(acc,indices='open')
    betax = twi0.betax
    betax = betax*(1/5)
    
    spos = _pyaccel.lattice.find_spos(acc)

    fig, (a1, a2, a3) = _plt.subplots(1, 3, figsize=(10, 5), sharey=True, gridspec_kw={'width_ratios': [1, 3, 3]})
    
    # defining the form that graphic will be plotted, trying to amplify the best I can the letters to see it in a easier way
    a1.grid(True, alpha=0.5, ls='--', color='k')
    a1.tick_params(axis='both', labelsize=12)
    a2.grid(True, alpha=0.5, ls='--', color='k')
    a2.tick_params(axis='both', labelsize=12)
    a3.grid(True, alpha=0.5, ls='--', color='k')
    a3.tick_params(axis='both', labelsize=12)
    a1.xaxis.grid(False)
    a2.xaxis.grid(False)
    a3.xaxis.grid(False)
    

    if 'pos' in param: # defining the y and x label of the first graphic
        a1.set_ylabel(r'positive $\delta$ [%]', fontsize=14)
        
        a3.plot(spos[int(lista_resul[1][-1])], lista_resul[2][-1]*1e2, 'r.', label='lost pos. (track)')
        acp_s = accep[1][element_idx] # defining the acceptance given the begining tracking point, this will be necessary to define until where the graphic will be plotted
        indx = _np.argmin(_np.abs(lista_off-acp_s))
        for item in lista_resul:
            a3.plot(spos[int(item[1])], item[2]*1e2, 'r.')
    elif'neg' in param:
        a1.set_ylabel(r'negative $\delta$ [%]', fontsize=14)
        a3.plot(spos[int(lista_resul[1][-1])], -lista_resul[2][-1]*1e2, 'r.', label='lost pos. (track)')
        acp_s = accep[0][element_idx] # defining the acceptance given the begining tracking point, this will be necessary to define until where the graphic will be plotted
        indx = _np.argmin(_np.abs(lista_off-acp_s))
        for item in lista_resul:
            a3.plot(spos[int(item[1])], -item[2]*1e2, 'r.')
            
    a1.set_title(r'taxa de espalhamento touschek', fontsize=14) # setting the title of the first graphic
    a1.set_xlabel(r'Scattering touschek rate', fontsize=14)

    a1.plot(f_dens, delt, label='Scattering touschek rate', color='black')
    

    a2.set_title(r'$\delta \times$ lost turn', fontsize=16) # setting the tilte of the second graphic
    a2.set_xlabel(r'n de voltas', fontsize=14)
    for iten in lista_resul:
        a2.plot(iten[0], iten[2]*1e2, 'k.', label = '')

    
    a3.set_title(r'tracking ', fontsize=16) # setting the title of the third graphic
    a3.plot(spos[lista_idx][:indx], lista_off[:indx]*1e2,'b.', label=r'accep. limit', alpha=0.25)
    
    _plt.hlines(1e2*acp_s, spos[0], spos[-1], color='black', linestyles='dashed', alpha=0.5) # acceptance cutoff
    a3.plot(spos, _np.sqrt(betax),color='orange', label=r'$ \sqrt{\beta_x}  $') # beta function
    _pyaccel.graphics.draw_lattice(acc, offset=-0.5, height=0.5, gca=True) #magnetic lattice
    
    a3.plot(spos[element_idx], 0, 'ko', label='{}, ({} m)'.format(
        acc[element_idx].fam_name, "%.2f" % spos[element_idx])) # initial position where tracking begins
    
    a3.set_xlabel(r'$s$ [m]', fontsize=14) # setting configurations of the graphic
    a3.legend(loc='best', ncol=2)

    fig.tight_layout()
    fig.show()
    
def select_idx(list_, param1, param2):
    arr = _np.array(list_)
    
    n_arr = arr[param1:param2+1]
    
    return n_arr

def t_list(elmnt):
    #this condition significates that if the input is only a number, then 
    #the fucntion transforms it into a list to avoid errors. Actually, I will delete this function,
    # so just forget this bulshit 
    if type(elmnt) == float or type(elmnt) ==  int:
        return [elmnt]
    else:
        return list(elmnt)
        

def f_function_arg_mod(kappa, kappam, b1_, b2_, norm):

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

    if norm:
        pass

    else:
        arg *= 2*_np.sqrt(_np.pi*(b1_**2-b2_**2))*taum

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

def norm_d(acc, lsps, scalc,_npt, norm=True):

    spos = _pyaccel.lattice.find_spos(acc, indices='closed')
    beta = _beam_rigidity(energy=3)[2]
    ltime = _pyaccel.lifetime.Lifetime(acc)
    b1, b2 = ltime.touschek_data['touschek_coeffs']['b1'],ltime.touschek_data['touschek_coeffs']['b2']

    calc_dp, calc_dn = [], []
    deltasp, deltasn = [], []
    indices, indices_model= [], []

    for _, s in enumerate(lsps):
        
        idx = _np.argmin(_np.abs(scalc - s))
        indices.append(idx)
        idx_model = _np.argmin(_np.abs(spos - s))
        indices_model.append(idx_model)
        
        kappam_p0 = 0.00001 # teste sugerido pelo ximenes
        kappam_n0 = 0.00001

        kappap = _np.linspace(kappam_p0, _np.pi/2, _npt)
        deltap = 1/beta * _np.tan(kappap)
        kappan = _np.linspace(kappam_n0, _np.pi/2, _npt)
        deltan = 1/beta * _np.tan(kappan)

        y_p = f_function_arg_mod(kappa=kappap,kappam=kappam_p0,b1_=b1[idx],b2_=b2[idx], norm=norm).squeeze()
        y_n = f_function_arg_mod(kappa=kappan,kappam=kappam_n0,b1_=b1[idx],b2_=b2[idx], norm=norm).squeeze()
        norm_facp = _scyint.trapz(y_p, deltap)
        norm_facn = _scyint.trapz(y_n, deltan)

#         Normalizing to obtain the probability density function

        y_p /= (norm_facp)
        y_n /= (norm_facn)

        calc_dp.append(y_p)
        calc_dn.append(y_n)
        deltasp.append(deltap)
        deltasn.append(deltan)
        
    calc_dp = _np.array(calc_dp)
    calc_dn = _np.array(calc_dn)
    deltasp = _np.array(deltasp)
    deltasn = _np.array(deltasn)
    indices = _np.array(indices)
    indices_model = _np.array(indices_model)

    return calc_dp, calc_dn, deltasp, deltasn, indices, indices_model

def get_scaccep(acc, accep):
    spos = _pyaccel.lattice.find_spos(acc, indices='closed')

    npt = int((spos[-1]-spos[0])/0.1)
    scalc = _np.linspace(spos[0],spos[-1], npt)
    daccpp = _np.interp(scalc, spos, accep[1])
    daccpn = _np.interp(scalc, spos, accep[0])

    return scalc, daccpp, daccpn
    
def n_norm_d(acc, lsps, _npt, getsacp, cutoff, norm=False):

    scalc, daccpp, daccpn = getsacp
    beta = _beam_rigidity(energy=3)[2]

    taum_p = (beta*daccpp)**2
    taum_n = (beta*daccpn)**2
    kappam_p = _np.arctan(_np.sqrt(taum_p))
    kappam_n = _np.arctan(_np.sqrt(taum_n))
    
    ltime = _pyaccel.lifetime.Lifetime(acc)
    b1, b2 = ltime.touschek_data['touschek_coeffs']['b1'],ltime.touschek_data['touschek_coeffs']['b2']

    calc_dp, calc_dn = [], []
    deltasp, deltasn = [], []

    for _, s in enumerate(lsps):
        
        idx = _np.argmin(_np.abs(scalc - s))

        kappam_p0 = kappam_p[idx]
        kappam_n0 = kappam_n[idx]
        
        kappam_p0x = cutoff # teste sugerido pelo Ximenes
        kappam_n0x = cutoff

        kappap = _np.linspace(kappam_p0x, _np.pi/2, _npt)
        deltap = 1/beta * _np.tan(kappap)
        kappan = _np.linspace(kappam_n0x, _np.pi/2, _npt)
        deltan = 1/beta * _np.tan(kappan)

        y_p = f_function_arg_mod(kappa=kappap,kappam=kappam_p0x,b1_=b1[idx],b2_=b2[idx], norm=norm).squeeze()
        y_n = f_function_arg_mod(kappa=kappan,kappam=kappam_n0x,b1_=b1[idx],b2_=b2[idx], norm=norm).squeeze()
        indp = _np.argmin(_np.abs(deltap- 1/beta * _np.tan(kappam_p0)))
        indn = _np.argmin(_np.abs(deltan- 1/beta * _np.tan(kappam_n0)))

        calc_dp.append(y_p[indp:])
        calc_dn.append(y_n[indn:])
        deltasp.append(deltap[indp:])
        deltasn.append(deltan[indn:])

    calc_dp = _np.array(calc_dp, dtype=object)
    calc_dn = _np.array(calc_dn, dtype=object)
    deltasp = _np.array(deltasp, dtype=object)
    deltasn = _np.array(deltasn, dtype=object)

    return calc_dp, calc_dn, deltasp, deltasn

# Como discutido no dia 23.08.2023 a primeira abordagem para a realizaçao da pesagem vai 
# vai ser feita definindo o corte como sendo a aceitancia de energia, isso foi deinido com base
# no cálculo já implementados para o tempo de vida touschek que é coerente com o tempo do SIRIUS
# futuramente isso pode ser alterado e redefinido

def nnorm_cutacp(acc, lsps, _npt, getsacp, norm=False):

    scalc, daccpp, daccpn = getsacp
    beta = _beam_rigidity(energy=3)[2]

    taum_p = (beta*daccpp)**2
    taum_n = (beta*daccpn)**2
    kappam_p = _np.arctan(_np.sqrt(taum_p))
    kappam_n = _np.arctan(_np.sqrt(taum_n))
    
    ltime = _pyaccel.lifetime.Lifetime(acc)
    b1, b2 = ltime.touschek_data['touschek_coeffs']['b1'],ltime.touschek_data['touschek_coeffs']['b2']

    calc_dp, calc_dn = [], []
    deltasp, deltasn = [], []

    for _, s in enumerate(lsps):
        
        idx = _np.argmin(_np.abs(scalc - s))
        kappam_p0 = kappam_p[idx]
        kappam_n0 = kappam_n[idx]

        kappap = _np.linspace(kappam_p0, _np.pi/2, _npt)
        deltap = 1/beta * _np.tan(kappap)
        kappan = _np.linspace(kappam_n0, _np.pi/2, _npt)
        deltan = 1/beta * _np.tan(kappan)

        y_p = f_function_arg_mod(kappa=kappap,kappam=kappam_p0,b1_=b1[idx],b2_=b2[idx], norm=norm).squeeze()
        y_n = f_function_arg_mod(kappa=kappan,kappam=kappam_n0,b1_=b1[idx],b2_=b2[idx], norm=norm).squeeze()
        
        # eliminating the negative values from array
        indp = _np.where(y_p<0)[0]
        indn = _np.where(y_n<0)[0]
        y_p[indp] = 0
        y_n[indn] = 0
        
        calc_dp.append(y_p)
        calc_dn.append(y_n)
        deltasp.append(deltap)
        deltasn.append(deltan)

    calc_dp = _np.array(calc_dp, dtype=object)
    calc_dn = _np.array(calc_dn, dtype=object)
    deltasp = _np.array(deltasp, dtype=object)
    deltasn = _np.array(deltasn, dtype=object)

    return calc_dp, calc_dn, deltasp, deltasn


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

def defining_tables():

    return None