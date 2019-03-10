#!/usr/bin/env bash

# --- TB ---
# ./run_cycle.py 'TB-Fam:PS-B' &
./sirius-script-ma-cycling.py 'TB-01:PS-QD1' &
./sirius-script-ma-cycling.py 'TB-01:PS-QF1' &
./sirius-script-ma-cycling.py 'TB-02:PS-QD2A' &
./sirius-script-ma-cycling.py 'TB-02:PS-QF2A' &
./sirius-script-ma-cycling.py 'TB-02:PS-QD2B' &
./sirius-script-ma-cycling.py 'TB-02:PS-QF2B' &
./sirius-script-ma-cycling.py 'TB-03:PS-QD3' &
./sirius-script-ma-cycling.py 'TB-03:PS-QF3' &
./sirius-script-ma-cycling.py 'TB-04:PS-QD4' &
./sirius-script-ma-cycling.py 'TB-04:PS-QF4' &
./sirius-script-ma-cycling.py 'TB-01:PS-CH-1' &
./sirius-script-ma-cycling.py 'TB-01:PS-CV-1' &
./sirius-script-ma-cycling.py 'TB-01:PS-CH-2' &
./sirius-script-ma-cycling.py 'TB-01:PS-CV-2' &
./sirius-script-ma-cycling.py 'TB-02:PS-CH-1' &
./sirius-script-ma-cycling.py 'TB-02:PS-CV-1' &
./sirius-script-ma-cycling.py 'TB-02:PS-CH-2' &
./sirius-script-ma-cycling.py 'TB-02:PS-CV-2' &
./sirius-script-ma-cycling.py 'TB-04:PS-CH' &
./sirius-script-ma-cycling.py 'TB-04:PS-CV-1' &
./sirius-script-ma-cycling.py 'TB-04:PS-CV-2' &
# --- BO ---
# ./run_cycle.py 'BO-Fam:PS-QF' &
# ./run_cycle.py 'BO-Fam:PS-QD' &
# ./run_cycle.py 'BO-Fam:PS-SF' &
# ./run_cycle.py 'BO-Fam:PS-SD' &
# ./run_cycle.py 'BO-Fam:PS-B-2' &
