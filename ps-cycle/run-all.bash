#!/usr/bin/env bash

./run_cycle


./run_cycle 'BO-Fam:PS-QF' &
./run_cycle 'BO-Fam:PS-QD' &
./run_cycle 'BO-Fam:PS-SF' &
./run_cycle 'BO-Fam:PS-SD' &
./run_cycle 'BO-Fam:PS-B-2' &
./run_cycle 'TB-Fam:PS-B' &
./run_cycle 'TB-01:PS-QD1' &
./run_cycle 'TB-01:PS-QF1' &
./run_cycle 'TB-02:PS-QD2A' &
./run_cycle 'TB-02:PS-QF2A' &
./run_cycle 'TB-02:PS-QD2B' &
./run_cycle 'TB-02:PS-QF2B' &
./run_cycle 'TB-03:PS-QD3' &
./run_cycle 'TB-03:PS-QF3' &
./run_cycle 'TB-04:PS-QD4' &
./run_cycle 'TB-04:PS-QF4' &
./run_cycle 'TB-01:PS-CH-1' &
./run_cycle 'TB-01:PS-CV-1' &
./run_cycle 'TB-01:PS-CH-2' &
./run_cycle 'TB-01:PS-CV-2' &
./run_cycle 'TB-02:PS-CH-1' &
./run_cycle 'TB-02:PS-CV-1' &
./run_cycle 'TB-02:PS-CH-2' &
./run_cycle 'TB-02:PS-CV-2' &
./run_cycle 'TB-04:PS-CH' &
./run_cycle 'TB-04:PS-CV-1' &
./run_cycle 'TB-04:PS-CV-2' &
