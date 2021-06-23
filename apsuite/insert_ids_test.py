from apsuite.insertion_devices.id_study import IDParams, ID

params = IDParams(IDParams.Beamlines.MANACA)
idclass = ID(params)

print(idclass.bare_model.length)
print(idclass.bare_model[2587])

mod_manaca = idclass.insert_ids()
print(mod_manaca.length)
print(mod_manaca[2587])

mod_manaca = idclass.fix_tunes(mod_manaca)
mod_manaca = idclass.symmetrize_straight_section(mod_manaca)
