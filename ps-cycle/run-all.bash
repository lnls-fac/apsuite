#!/usr/bin/env bash

# --- TB ---
./run_cycle.py 'TB-Fam:PS-B' &
./run_cycle.py 'TB-01:PS-QD1' &
./run_cycle.py 'TB-01:PS-QF1' &
./run_cycle.py 'TB-02:PS-QD2A' &
./run_cycle.py 'TB-02:PS-QF2A' &
./run_cycle.py 'TB-02:PS-QD2B' &
./run_cycle.py 'TB-02:PS-QF2B' &
./run_cycle.py 'TB-03:PS-QD3' &
./run_cycle.py 'TB-03:PS-QF3' &
./run_cycle.py 'TB-04:PS-QD4' &
./run_cycle.py 'TB-04:PS-QF4' &
./run_cycle.py 'TB-01:PS-CH-1' &
./run_cycle.py 'TB-01:PS-CV-1' &
./run_cycle.py 'TB-01:PS-CH-2' &
./run_cycle.py 'TB-01:PS-CV-2' &
./run_cycle.py 'TB-02:PS-CH-1' &
./run_cycle.py 'TB-02:PS-CV-1' &
./run_cycle.py 'TB-02:PS-CH-2' &
./run_cycle.py 'TB-02:PS-CV-2' &
./run_cycle.py 'TB-04:PS-CH' &
./run_cycle.py 'TB-04:PS-CV-1' &
./run_cycle.py 'TB-04:PS-CV-2' &
# --- BO ---
./run_cycle.py 'BO-Fam:PS-QF' &
./run_cycle.py 'BO-Fam:PS-QD' &
./run_cycle.py 'BO-Fam:PS-SF' &
./run_cycle.py 'BO-Fam:PS-SD' &
./run_cycle.py 'BO-Fam:PS-B-2' &
