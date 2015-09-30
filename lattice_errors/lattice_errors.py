
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
                 as segments of the same physical and the same errors will be
                 applied to them.
                 Its default is None, which means each instance of the magnets in
                 the lattice will be considered as independent, with its own error
                 and girder errors will be ignored.
                 For sirius, this dictionary can be created with the function
                 sirius.<'version'>.get_family_data(acc).
      nr_mach : generate errors for this number of machines.
      cutoff  : number of sigmas to truncate the distribution (default is infinity)
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
