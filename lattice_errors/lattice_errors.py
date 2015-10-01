
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


# def calc_respm_cod(acc,bpm_idx,hcm_idx,vcm_idx,symmetry=1,printing=False):
#     """
#
#     """
#
#     def get_response_matrix(bpms, hcms, vcms):
#         # M(y,x) --> y : orbit    x: corrector
#         M, T = _pyaccel.tracking.findm66(acc, 1:length(the_ring)+1);
#
#
#         len_hcms = len(hcms)
#         len_vcms = len(vcms)
#         len_bpms = len(bpms)
#
#         mxx = _np.zeros((len_bpms, len_hcms))
#         myx = _np.zeros((len_bpms, len_hcms))
#         mxy = _np.zeros((len_bpms, len_vcms))
#         myy = _np.zeros((len_bpms, len_vcms))
#
#         len_hcms = [sum([acc[j].length for j in hcms[i]]) for i in range(len(hcms))]
#         len_vcms = [sum([acc[j].length for j in vcms[i]]) for i in range(len(vcms))]
#
#         D = diag(ones(1,size(M,1)));
#         for i=1:len_bpms
#             R_i = T(:,:,bpms(i,end));
#             DM_i = (D - R_i * M / R_i);
#             for j=1:len_hcms
#                 [mxx(i,j), myx(i,j), ~, ~] = get_C(T,DM_i, R_i, bpms(i,:),hcms(j,:),total_hcms(j));
#
#             for j=1:len_vcms
#                 [~, ~, mxy(i,j), myy(i,j)] = get_C(T,DM_i, R_i, bpms(i,:),vcms(j,:),total_vcms(j));
#
#
#         MR = [mxx, mxy; myx, myy];
#
#         def get_C(T,DM_i, R_i, i,j, length):
#             # cxy --> orbit at bpm x due to kick in corrector y
#
#             #R_j = T(:,:,j(end));
#             if (i(end)>j(end))
#                 R_ij = R_i/T(:,:,j(end)); % R_i/R_j
#             else
#                 R_ij = R_i * (T(:,:,end) / T(:,:,j(end)));
#
#             #C = R_ij / (diag([1 1 1 1])-M44);
#             C = DM_i \ R_ij
#
#             cxx = -(length/2)*C[0,0]   +   C[0,1]
#             cyx = -(length/2)*C[2,0]   +   C[2,1]
#             cxy = -(length/2)*C[0,2]   +   C[0,3]
#             cyy = -(length/2)*C[2,2]   +   C[2,3]
#
#
#     # making sure they are in order
#     bpm_idx = sorted(bpm_idx)
#     hcm_idx = sorted(hcm_idx)
#     vcm_idx = sorted(vcm_idx)
#
#
#     nr_bpms = len(bpm_idx)
#     nr_hcms = len(hcm_idx)
#     nr_vcms = len(vcm_idx)
#
#     if printing:
#         print('bpms:{0:03d}, hcms:{0:03d}, vcms:{0:03d}'.format(
#                    nr_bpms, nr_hcms, nr_vcms))
#
#
#     Mxx = _np.zeros(nr_bpms, nr_hcms)
#     Myx = _np.zeros(nr_bpms, nr_hcms)
#     Mxy = _np.zeros(nr_bpms, nr_vcms)
#     Myy = _np.zeros(nr_bpms, nr_vcms)
#
#     len_bpm = nr_bpms // symmetry
#     len_hcm = nr_hcms // symmetry
#     len_vcm = nr_vcms // symmetry
#     if (nr_bpm % 2) or (nr_hcm % 2) or (len_vcm % 2):
#         len_bpm = nr_bpm
#         len_hcm = nr_hcm
#         len_vcm = nr_vcm
#         symmetry = 1
#     else:
#         hcm_idx = hcm_idx[:len_hcm]
#         vcm_idx = vcm_idx[:len_vcm]
#
#
#     M = get_response_matrix(the_ring, bpm_idx, hcm_idx, vcm_idx);
#     M = mat2cell(M, [nr_bpms,nr_bpms],[len_hcm, len_vcm]);
#     mxx = M{1,1};
#     mxy = M{1,2};
#     myx = M{2,1};
#     myy = M{2,2};
#
#     for i=0:(symmetry-1)
#         indcs = (i*len_hcm+1):((i+1)*len_hcm);
#         Mxx(:,indcs) = circshift(mxx,len_bpm*i);
#         Myx(:,indcs) = circshift(myx,len_bpm*i);
#
#         indcs = (i*len_vcm+1):((i+1)*len_vcm);
#         Mxy(:,indcs) = circshift(mxy,len_bpm*i); %the last bpm turns into the first
#         Myy(:,indcs) = circshift(myy,len_bpm*i);
#     end
#
#     r.respm.mxx = Mxx;
#     r.respm.mxy = Mxy;
#     r.respm.myx = Myx;
#     r.respm.myy = Myy;
#
#     [U,S,V] = svd([Mxx Mxy; Myx Myy],'econ');
#     r.respm.U = U;
#     r.respm.V = V;
#     r.respm.S = S;
#
#     sv = diag(S);
#     if print
#         fprintf('   number of singular values: %03i\n', size(S,1));
#         fprintf('   singular values: %f,%f,%f ... %f,%f,%f\n', sv(1),sv(2),sv(3),sv(end-2),sv(end-1),sv(end));
#     end
