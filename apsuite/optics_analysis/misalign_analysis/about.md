# Pynel package for vertical dispersion analysis

The pynel package has 2 main objects: Base and Button. These objects have properties that envolves the analysis of vertical dispersion function of the SIRIUS storage ring and its signatures associated to magnets tranversal and rotation misalignments.

## Object Button

The Button object associates one kind of error (transversal or rotation misalignment: x, y, roll, pitch and yaw) to one magnet of the SIRIUS ring and store the vertical dispersion signature caused by the magnet and the error choosen.

#### The creation
The creation of a Button follows 2 possible ways. 
- 1st. Passing 3 arguments: the magnet name, the magnet sector and the error associated. \
Example: ``` qfa_sect5_dx = Button(sect=5, name="QFA", dtype="dx")``` -> creates a quadrupole QFA Button located in the 5th sector with tranversal horizontal misalignment error. 
- 2nd. Passing 2 arguments: the magnet indices in the SIRIUS _"pymodels"_ model, and the error associated. \
Example: ``` sfa1_sect1_dr = Button(indices=[74], dtype="dx")``` -> creates a sextupole SFA1 Button located in the 1st sector with rotation roll error. 
For the 2nd option to create a Button, its necessary to check if Pymodels is up-to-date.

#### About the arguments
- ```name```: the creation of any Button requires its name (when not passing "indices" arg) that is any magnet family name that exists in the SIRIUS ring: _"B1, B2, BC, Q1, Q2 ... QFA ... QDB2 ... SFA1 ... SFP2 ... SDB3"_. \
If a inexistent family name is passed (or any other random thing) the Button will still be generated, but invalid, that means the Button will not compute and/or store a Vertical Dispersion Signature. \
Obs.: In sectors that have more then one magnet of the same family (like dipoles B1 and B2 or quadupoles Q1, Q2...) the ```name``` argument can specify the precise magnet wanted: ```b1_1_sect7_dy = Button(7, "B1_1", "dy")```.

- ```sect```: the creation of any Button requires its sector (when not passing "indices" arg). The sectors are restrict to integer numbers from 1 to 20 (the real sectors in SIRIUS). \
Like the ```name``` argument, if a non-existent sector is passed the Button will be generated, but invalid.

- ```indices```: the creation of any Button can be made by passing the indices of the magnet in the model. Is recommended to be careful when creating dipoles Buttons by its indices or when working with a refined model (non single integer as its indices).

- ```dtype```: the creation of any Button always requires its error associated. The error are restricted to: 
    - ```dx``` - Horizontal misalignment error
    - ```dy``` - Vertical misalignment error
    - ```dr``` - Rotation roll error
    - ```drp``` - Rotation pitch error
    - ```dry``` - Rotation yaw error 

#### About the properties
- ```.bname``` = the magnet family (button) name
- ```.fantasy_name``` = the magnet specified name
- ```.sect``` = the magnet sector location
- ```.sectype``` = the magnet sector type (like: "HighBetaA to LowBetaB") 
- ```.dtype``` = the error associated
- ```.signature``` = the vertical dispersion signature of the magnet with the error associated
- ```.indices```  = the indices of the magnet in the model

## Object Base
The Base object generate Buttons and construct a VDRM matrix (Vertical Dispersion Response Matrix) of the signatures of the buttons. Base objects can be use to easily creates sets of specified kind of magnets or errors or sectors and study with more detail the behavior of the Vertical Dispersion in these combinations.

#### The creation
The creation of a Base follows 2 possible ways. 
- 1st. Passing 3 arguments: the elements (magnets names/families), the sectors and the errors. \
Example: ```base = Base(sects=[1,7,13,20], elements=["SDB3", "BC"], dtypes=["dy", "dr"])``` -> creates a Base with the dipoles BC and sextupoles SDB3 in the sectors 1, 7, 13 and 20 with the vertical misalignment and rotation roll errors. \
Obs.: if any magnet not exists in the sector, the button will be discarded. (Except when controlling the ```default_valids``` arg)
- 2nd. Passing 1 arg: already generated buttons. Example: ``` base = Base(buttons=list_of_buttons)``` -> creates a Base with the buttons of the ```list_of_buttons```.

#### About the arguments
- ```elements```: the magnets family/fantasy names -> The valid "elements" follows the Buttons valid "names"/```bnames```
- ```sects``` : list of integers. The valid sectors are the integers from 1 to 20.
- ```dtypes``` : the misalignment/rotation errors. The valid "dtypes" follows the Buttons valid ```dtype```(s)
- ```buttons``` : should be a list of Button objects
- ```default_valids```: allows to control wether Button is considered valid or invalid. And if the Button will be or not discarded in the gen process.
- ```force_rebuild``` : force the calculation of the vertical dispersion signature of the buttons
- ```func```: specify the function to be calculated as the buttons signatures. valid "funcs" are: `vertical_disp` or `testfunc`. The "testfunc" simply sets the buttons signatures as a zero-arrays (used for sandbox creation of buttons)

#### About the properties
- ```.magnets``` = the buttons magnet families
- ```.named_magnets``` = the buttons magnet fantasy (specified) names
- ```.sectors``` = the sectors of the Base
- ```.sector_types``` = the sector types (like: "HighBetaA to LowBetaB") of the sectors of the Base
- ```.dtypes``` = the errors of the Base
- ```.buttons``` = the buttons of the Base
- ```.resp_mat``` = the constructed VDRM (Vertical Dispersion Response Matrix)

## The "fitting" module

The fitting module contains functions to fit vertical dispertion function (like real data collected in the machine) in _pymodels_ models.
Obs.: this module is outdated (~ september 19, 2023). The functions shouldnt work as expected.

## The "misc_functions"/"functions" module

Contains functions to work with Base and Buttons objects and deal with vertical dispertion fittings and analysis.

## The "std_si_Data" module

Contains saved data of the Standard SIRIUS model.