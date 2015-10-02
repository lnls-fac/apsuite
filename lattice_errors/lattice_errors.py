
import numpy as _np
import matplotlib.pyplot as _plt
import scipy.stats as _scystat

import pyaccel as _pyaccel
import mathphys as _mp

_ERRORTYPES = ('x','y','roll','yaw','pitch','excit','k_dip')

def generate_errors(name,acc,config,fam_data=None,nr_mach=20,cutoff=1,rndtype='gauss'):
    """Generates random errors to be applied in the model by the function apply_errors.

    INPUTS:
      name       : a string to be used as name to save the configuration data
      acc        : is the accelerator model.
      config     : dictionary with configuration of the errors with keys
                   'mags' and/or 'girders'. All of them are optional:
        'mags'   : dictionary with arbitrary named keys whose values are
                   dictionaries with keys:
          'labels': list of family names
          'sigma' : dictionary. possible keys:'x','y','roll','excit','k_dip','yaw','pitch'.
                    Values are errors definition (1-sigma for gauss, max for uniform)

        'girder' : dictionary with definition of girder errors. Possible keys:
                   'x','y','roll','excit','k_dip','yaw','pitch'.
                   Values are errors definition (1-sigma for gauss, max for uniform)
      fam_data : dictionary whose keys are family names of, at least, all magnets
                 defined in config['mags']['labels'] and the key 'girder' and
                 values are dictionaries with at least one key: 'index',
                 whose value is a list of indices the magnets in the lattice.
                 If this list is a nested list, then each sub-list is understood
                 as segments of the same physical magnet and the same error will be
                 applied to them.
                 Its default is None, which means each instance of the magnets in
                 the lattice will be considered as independent, with its own error
                 and girder errors will be ignored.
                 For sirius, this dictionary can be created with the function
                 sirius.<'version'>.get_family_data(acc).
      nr_mach : generate errors for this number of machines.
      cutoff  : number of sigmas to truncate the distribution (default is 1)
      rndtype : type of distribution. Possible values: 'uniform' and 'gauss'.
                Default is 'gauss'.

    OUTPUT:
        errors : dictionary with keys: 'x','y','roll','yaw','pitch','excit','k_dip'.
          Each key is a list with dimension nr_mach x len(acc)
          with errors generated for the elements defined by the inputs. If an
          element errors has contributions from 'mags' and 'girder', the value
          present in this output will be the sum of them.

    EXAMPLES:
     >>> acc = sirius.si.SI_V10.create_accelerator()
     >>> fam_data = sirius.SI_V10.get_family_data(acc)
     >>> um, mrad, percent = 1e-6, 1e-3, 1e-2
     >>> config = dict({'mags':dict(),'girder':dict()})
     >>> config['mags']['quads'] = dict({'labels':list(),'sigma':dict()})
     >>> config['mags']['quads']['labels'] += ['qfa','qdb2','qfb']
     >>> config['mags']['quads']['sigma']['x']     = 40 * um * 1
     >>> config['mags']['quads']['sigma']['y']     = 40 * um * 1
     >>> config['mags']['quads']['sigma']['roll']  = 0.20 * mrad * 1
     >>> config['mags']['quads']['sigma']['excit'] = 0.05 * percent * 1
     >>> config['mags']['dips'] = dict({'labels':list(),'sigma':dict()})
     >>> config['mags']['dips']['labels'] += ['b1','b2']
     >>> config['mags']['dips']['sigma']['x']     = 40 * um * 1
     >>> config['mags']['dips']['sigma']['y']     = 40 * um * 1
     >>> config['girder']['x']     = 100 * um * 1
     >>> config['girder']['y']     = 100 * um * 1
     >>> config['girder']['roll']  =0.20 * mrad * 1
     >>> config['girder']['yaw']   =  20 * mrad * 0
     >>> config['girder']['pitch'] =  20 * mrad * 0
     >>> errors = generate_errors('test',acc,config,fam_data,nr_mach=20,cutoff=2)
    """

    #define the random numbers generator
    if rndtype.lower().startswith('gauss'):
        random_numbers = _scystat.truncnorm(-cutoff,cutoff).rvs
    elif rndtype.lower().startswith('unif'):
        random_numbers = _scystat.uniform(loc=-1,scale=2).rvs
    else:
        raise TypeError('Distribution type not recognized.')

    #generate empty arrays to store errors
    errors = dict()
    for errtype in _ERRORTYPES:
        errors[errtype] = _np.zeros((nr_mach, len(acc)))


    # _mp.utils.save_pickle(name+'_generate_errors_input',
    #           config=config,nr_mach=nr_mach,cutoff=cutoff,rndtype=rndtype)

    if 'mags' in config:
        for mtype in config['mags']:
            for errtype in config['mags'][mtype]['sigma']:
                for fam_name in config['mags'][mtype]['labels']:
                    if fam_data is not None:
                        idx = fam_data[fam_name]['index']
                    else:
                        idx = _pyaccel.lattice.find_indices(acc,'fam_name',fam_name)
                    idx = _np.array(idx)
                    rnd = random_numbers((nr_mach,len(idx)))
                    if isinstance(idx[0],(list,tuple,_np.ndarray)):
                        rnd = rnd.repeat(len(idx[0]),axis=1)
                    errors[errtype][:,idx.ravel()] += rnd * config['mags'][mtype]['sigma'][errtype]

    if ('girder' in config) and (fam_data is not None):
        for errtype in config['girder']:
            for gir_name in fam_data['girder']:
                idx = _np.array(fam_data['girder'][gir_name]['index'])
                rnd = random_numbers((nr_mach,1)).repeat(len(idx),axis=1)
                errors[errtype][:,idx.ravel()] += rnd * config['girder'][errtype]

    return errors

def apply_erros(name, machine, errors, increment=1.0):
    """Apply the errors generated by generate_errors to the ring model.

 INPUTS:
   name     : name of the file to save input parameters
   machine  : might be a model of the ring or a list of models of
              the ring
   errors   : structure of errors to be applied (for more details see
              generate_errors help
   increment: float defining the fraction of the errors which will be
              additively applied to the machines.

 OUTPUT:
   machine  : lis of ring models with errors.
"""
    def apply_errors_one_machine(ring, errors, ii, fraction):
        funs={'x':_pyaccel.lattice.add_error_misalignment_x,
              'y':_pyaccel.lattice.add_error_misalignment_y,
              'roll':_pyaccel.lattice.add_error_rotation_roll,
              'yaw':_pyaccel.lattice.add_error_rotation_yaw,
              'pitch':_pyaccel.lattice.add_error_rotation_pitch,
              'excit':_pyaccel.lattice.add_error_excitation_main,
              'k_dip':_pyaccel.lattice.add_error_excitation_kdip}

        for errtype in errors:
            err = fraction * errors[errtype][ii,:]
            idx, *_ = err.nonzero()
            funs[errtype](ring, idx, err[idx])

    nr_mach = errors['x'].shape[0]

    # _mp.save_pickle([name,'_apply_errors_input'],
    #                 errors=errors, increment=increment)

    machs = []
    if not isinstance(machine,list):
        for i in range(nr_mach):
            machs.append(_pyaccel.accelerator.Accelerator(accelerator=machine))
    machine = machs

    if len(machine) != nr_mach:
        print('DifferentSizes: Incompatibility between errors and'+
                ' machine lengths.\n Using minimum of both.')
        nr_mach = min([len(machine),nr_mach])

    ids_idx  = {i for i in range(len(machine[0]))
                  if machine[0][i].pass_method.startswith('kicktable_pass')}
    sext_idx = {i for i in range(len(machine[0]))
                  if machine[0][i].polynom_b[2] != 0.0}

    print('    ------------------------------- ')
    print('   |   codx [mm]   |   cody [mm]   |')
    print('   | (max)   (rms) | (max)   (rms) |')
    print('---|-------------------------------|')
   #print('001| 13.41   14.32 | 13.41   14.32 |');
    for i in range(nr_mach):
        apply_errors_one_machine(machine[i], errors, i, increment)
        ring = machine[i][:]
        for ii in range(len(ring)):
            if ii in sext_idx: ring[ii].polynom_b[2] = 0.0
            if ii in ids_idx : ring[ii].pass_method = 'drift_pass'
        codx, cody = _calc_cod(ring)
        x_max_all, x_rms_all = 1e3*_np.abs(codx).max(), 1e3*codx.std(ddof=1)
        y_max_all, y_rms_all =  1e3*_np.abs(cody).max(), 1e3*cody.std(ddof=1)
        print('{0:03d}| {1:5.2f}   {2:5.2f} | {3:5.2f}   {4:5.2f} |'.format(
                                    i,x_max_all,x_rms_all,y_max_all,y_rms_all))
    print(36*'-')
    return machine

def _calc_cod(ring):
    if ring.cavity_on:
        orb = _pyaccel.tracking.findorbit6(ring,indices='open')
    else:
        orb = _pyaccel.tracking.findorbit4(ring,indices='open')
    return orb[0], orb[2]


def lnls_latt_err_correct_cod(name, machine, orbit, goal_codx, goal_cody): return None
    # """
    # % Correct orbit of several machines.
    # %
    # % INPUTS:
    # %   name     : name of the file to which the inputs will be saved
    # %   machine  : cell array of lattice models to correct the orbit.
    # %   goal_codx: horizontal reference orbit to use in correction. May be a vector
    # %      defining the orbit for each bpm. In this case the reference will be the
    # %      same for all the machines. Or Can be a matrix with dimension
    # %      nr_machinesines X nr_bpms, to define different orbits among the machines.
    # %      If not passed a default of zero will be used;
    # %   goal_cody: same as goal_codx but for the vertical plane.
    # %   orbit    : structure with fields:
    # %      bpm_idx   - bpm indexes in the model;
    # %      hcm_idx   - horizontal correctors indexes in the model;
    # %      vcm_idx   - vertical correctors indexes in the model;
    # %      sext_ramp - If existent, must be a vector with components less than
    # %         one, denoting a fraction of sextupoles strengths used in each step
    # %         of the correction. For example, if sext_ramp = [0,1] the correction
    # %         algorithm will be called two times for each machine. In the first
    # %         time the sextupoles strengths will be zeroed and in the second
    # %         time they will be set to their correct value.
    # %      svs      - may be a number denoting how many singular values will be
    # %         used in the correction or the string 'all' to use all singular
    # %         values. Default: 'all';
    # %      max_nr_iter - maximum number of iteractions the correction
    # %         algortithm will perform at each call for each machine;
    # %      tolerance - if in two subsequent iteractions the relative difference
    # %         between the error function values is less than this value the
    # %         correction is considered to have converged and will terminate.
    # %      correct2bba_orbit - if true, the goal orbit will be set in relation
    # %         to the magnetic center of the quadrupole nearest to each bpm.
    # %      simul_bpm_err - if true, the Offset field defined in the bpms in the
    # %         lattice will be used to simulate an error in the determination of
    # %         the goal orbit. Notice that there must exist an Offset field defined
    # %         in the lattice models of the machine array for each bpm, in order to
    # %         this simulation work. Otherwise an error will occur.
    # %      respm - structure with fields M, S, V, U which are the response
    # %         matrix and its SVD decomposition. If NOT present, the function
    # %         WILL CALCULATE the response matrix for each machine in each step
    # %         defined by sext_ramp.
    # %
    # % OUTPUT:
    # %   machine : cell array of lattice models with the orbit corrected.
    # %
    # """
    #
    # # making sure they are in order
    # orbit['bpms'] = sorted(orbit['bpms'])
    # orbit['hcms'] = sorted(orbit['hcms'])
    # orbit['vcms'] = sorted(orbit['vcms'])
    #
    # nr_machines = len(machine)
    #
    # if goal_codx is None:
    #     goal_codx = _np.zeros(nr_machines,len(orbit['bpms']));
    # elif len(goal_codx) == 1:
    #     goal_codx = goal_codx*_np.ones(x,nr_machines,1);
    #
    # if goal_cody is None:
    #     goal_cody = _np.zeros(nr_machines,len(orbit['bpms']));
    # elif len(goal_cody) == 1:
    #     goal_cody = repmat(goal_cody,nr_machines,1);
    #
    #
    # if ~isfield(orbit,'sext_ramp'), orbit.sext_ramp = 1; end;
    #
    # save([name,'_correct_cod_input.mat'], 'orbit', 'goal_codx', 'goal_cody');
    #
    # calc_respm = false;
    # if ~isfield(orbit,'respm'), calc_respm = true; end
    #
    # print('-  correcting closed-orbit distortions\n');
    # print('   sextupole ramp: '); fprintf(' %4.2f', orbit.sext_ramp); fprintf('\n');
    # if isnumeric(orbit.svs), svs = num2str(orbit.svs);else svs = orbit.svs;end
    # print('   selection of singular values: %4s\n',svs);
    # print('   maximum number of orbit correction iterations: %i\n',orbit.max_nr_iter);
    # print('   tolerance: %8.2e\n', orbit.tolerance);
    #
    # print('\n');
    # print('    -----------------------------------------------------------------------------------------------  \n');
    # print('   |           codx [um]           |           cody [um]           |  kickx[urad]     kicky[urad]  | (nr_iter|nr_refactor)\n');
    # print('   |      all             bpm      |      all             bpm      |                               | [sextupole ramp]\n');
    # print('   | (max)   (rms) | (max)   (rms) | (max)   (rms) | (max)   (rms) | (max)   (rms) | (max)   (rms) | ');
    # print('%7.5f ', orbit.sext_ramp); fprintf('\n');
    # print('---|---------------------------------------------------------------|-------------------------------| \n');
    # if orbit.correct2bba_orbit
    #     ind_bba = get_bba_ind(machine{1});
    #
    #
    # sext_idx = findcells(machine{1},'PolynomB');
    # random_cod = zeros(2,length(orbit.bpm_idx));
    # ids_idx = findcells(machine{1}, 'PassMethod', 'LNLSThickEPUPass');
    #
    # for i=1:nr_machines
    #     sext_str = getcellstruct(machine{i}, 'PolynomB', sext_idx, 1, 3);
    #
    #     if orbit.simul_bpm_err
    #         random_cod = getcellstruct(machine{i},'Offsets',orbit.bpm_idx);
    #         random_cod = cell2mat(random_cod)';
    #     if orbit.correct2bba_orbit
    #         T1 = getcellstruct(machine{i},'T1',ind_bba,1,1);
    #         T2 = getcellstruct(machine{i},'T2',ind_bba,1,1);
    #         bba_codx = (T2-T1)'/2;
    #         T1 = getcellstruct(machine{i},'T1',ind_bba,1,3);
    #         T2 = getcellstruct(machine{i},'T2',ind_bba,1,3);
    #         bba_cody = (T2-T1)'/2;
    #     else
    #         bba_codx = 0;
    #         bba_cody = 0;
    #
    #     gcodx = goal_codx(i,:) + random_cod(1,:) + bba_codx;
    #     gcody = goal_cody(i,:) + random_cod(2,:) + bba_cody;
    #
    #     niter = zeros(1,length(orbit.sext_ramp));
    #     ntimes = niter;
    #
    #     for i in idx: the_ring{i}.PassMethod = 'DriftPass'
    #     for j=1:length(orbit.sext_ramp)
    #         if (j == length(orbit.sext_ramp))
    #             for i in idx: the_ring{i}.PassMethod = 'LNLSThickEPUPass'
    #         machine{i} = setcellstruct(machine{i},'PolynomB',sext_idx,orbit.sext_ramp(j)*sext_str, 1, 3);
    #         if calc_respm
    #             orbit.respm = calc_respm_cod(machine{i}, orbit.bpm_idx, orbit.hcm_idx, orbit.vcm_idx, 1, false);
    #             orbit.respm = orbit.respm.respm;
    #         [machine{i},hkck,vkck,codx,cody,niter(j),ntimes(j)] = cod_sg(orbit, machine{i}, gcodx, gcody);
    #         if any(isnan([codx,cody]))
    #             fprintf('Machine %03i unstable @ sextupole ramp = %5.1f %%\n',i,sextupole_ramp(j)*100);
    #             machine{i} = setcellstruct(machine{i},'PolynomB',sext_idx, sext_str, 1, 3);
    #             break;
    #
    #     x_max_all,x_rms_all = get_max_rms(codx,1e6);
    #     x_max_bpm,x_rms_bpm = get_max_rms(codx(orbit.bpm_idx),1e6);
    #     y_max_all,y_rms_all = get_max_rms(cody,1e6);
    #     y_max_bpm,y_rms_bpm = get_max_rms(cody(orbit.bpm_idx),1e6);
    #     kickx_max,kickx_rms = get_max_rms(hkck,1e6);
    #     kicky_max,kicky_rms = get_max_rms(vkck,1e6);
    #     print('%03i| %5.1f   %5.1f | %5.1f   %5.1f | %5.1f   %5.1f | %5.1f   %5.1f |  %3.0f     %3.0f  |  %3.0f     %3.0f  | ', i, ...
    #         x_max_all,x_rms_all,x_max_bpm,x_rms_bpm,y_max_all,y_rms_all,y_max_bpm,y_rms_bpm, ...
    #         kickx_max,kickx_rms,kicky_max,kicky_rms);
    #     print('(%02i|%02i) ', [niter(:) ntimes(:)]'); fprintf('\n');
    #
    # print('--------------------------------------------------------------------------------------------------- \n');
    #
    # def get_max_rms(v,f):
    #     maxv = f*max(abs(v));
    #     rmsv = f*sqrt(sum(v.^2)/length(v));
    #     return maxv, rmsv



def calc_respm_cod(acc,bpm_idx,hcm_idx,vcm_idx,symmetry=1,printing=False):
    """

    """

    def get_response_matrix(acc, bpms, hcms, vcms):
        # M(y,x) --> y : orbit    x: corrector

        M, T = _pyaccel.tracking.findm66(acc)

        A_InvB = lambda A,B: _np.linalg.solve(B.T, A.T).T
        InvA_B = lambda A,B: _np.linalg.solve(A, B)
        def get_C(DM_i, R0i, bpm, corr, length):
            # cxy --> orbit at bpm x due to kick in corrector y
            if bpm>corr:
                R_ij = A_invB(R0i,T[corr]) # R0i/R0j
            else :
                R_ij = _np.dot(R0i, A_InvB(M, T[corr])) # Rij = R0i*M*inv(R0j)

            C = inA_B(DM_i, R_ij)
            cxx = -(length/2)*C[0,0]   +   C[0,1]
            cyx = -(length/2)*C[2,0]   +   C[2,1]
            cxy = -(length/2)*C[0,2]   +   C[0,3]
            cyy = -(length/2)*C[2,2]   +   C[2,3]
            return cxx, cyx, cxy, cyy

        nr_hcms = len(hcms)
        nr_vcms = len(vcms)
        nr_bpms = len(bpms)
        mxx = _np.zeros((nr_bpms, nr_hcms))
        myx = _np.zeros((nr_bpms, nr_hcms))
        mxy = _np.zeros((nr_bpms, nr_vcms))
        myy = _np.zeros((nr_bpms, nr_vcms))
        len_hcms = [sum([acc[j].length for j in hcms[i]]) for i in range(len(hcms))]
        len_vcms = [sum([acc[j].length for j in vcms[i]]) for i in range(len(vcms))]
        D = _np.eye(M.shape[0])
        for i in range(nr_bpms):
            R_i = T[bpms[i][-1]]
            DM_i = D - A_invB(_np.dot(R_i,M), R_i) # I - R*M*inv(R)
            for j in range(nr_hcms):
                mxx[i,j], myx[i,j], *_ = get_C(DM_i, R_i, bpms[i][-1],hcms[j][-1],len_hcms[j])
            for j in range(nr_vcms):
                *_, mxy[i,j], myy[i,j] = get_C(DM_i, R_i, bpms[i][-1],vcms[j][-1],len_vcms[j])
        return mxx,mxy,myx,myy


    # making sure they are in order
    bpm_idx = sorted(bpm_idx)
    hcm_idx = sorted(hcm_idx)
    vcm_idx = sorted(vcm_idx)
    nr_bpms = len(bpm_idx)
    nr_hcms = len(hcm_idx)
    nr_vcms = len(vcm_idx)
    M = _np.zeros(2*nr_bpms,nr_vchms+nr_hcms) # create response matrix and its components
    Mxx, Mxy = M[:nr_bpms,:nr_hcms], M[:nr_bpms,nr_hcms:]
    Myx, Myy = M[nr_bpms:,:nr_hcms], M[nr_bpms:,nr_hcms:]

    len_bpm = nr_bpms // symmetry
    len_hcm = nr_hcms // symmetry
    len_vcm = nr_vcms // symmetry
    if (len_bpm % 1) or (len_hcm % 1) or (len_vcm % 1):
        len_bpm = nr_bpm
        len_hcm = nr_hcm
        len_vcm = nr_vcm
        symmetry = 1
    else:
        hcm_idx = hcm_idx[:len_hcm]
        vcm_idx = vcm_idx[:len_vcm]

    if printing:
        print('bpms:{0:03d}, hcms:{0:03d}, vcms:{0:03d}'.format(
                   nr_bpms, nr_hcms, nr_vcms))

    mxx,mxy,myx,myy = get_response_matrix(acc, bpm_idx, hcm_idx, vcm_idx)

    for i in range(symmetry):
        indcs = list(range(i*len_hcm,(i+1)*len_hcm))
        Mxx[:,indcs] = np.roll(mxx,len_bpm*i,axis=0)
        Myx[:,indcs] = np.roll(myx,len_bpm*i,axis=0)

        indcs = list(range(i*len_vcm,(i+1)*len_vcm))
        Mxy[:,indcs] = np.roll(mxy,len_bpm*i,axis=0) #the last bpm turns into the first
        Myy[:,indcs] = np.roll(myy,len_bpm*i,axis=0)


    r['M'] = M

    U, s, V = np.linalg.svd(M,full_matrices=False) #M = U*np.diag(s)*V
    r['U'] = U
    r['V'] = V
    r['s'] = s

    if printing:
        print('number of singular values: {0:03d}'.format(len(s)))
        print('singular values: {0:3f},{0:3f},{0:3f} ... {0:3f},{0:3f},{0:3f}'.format(
                                 s[0],  s[1],  s[2],      s[-3],s[-2],s[-1]))
