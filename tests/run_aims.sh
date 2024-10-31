#!/usr/bin/env bash

# Parse args
binary=$1
run_aims=$2

if [[ -f "tests/fixtures/custom_bin_aims_calcs/1/aims.out" && "$run_aims" == "change_bin" ]]; then
    rm -rf fixtures/custom_bin_aims_calcs
elif [[ -f "tests/fixtures/custom_bin_aims_calcs/1/aims.out" && "$run_aims" == "True" ]]; then
    exit 1
fi

for i in {1..10}; do (
    # Copy the aims input files to a new directory
    mkdir -p tests/fixtures/custom_bin_aims_calcs/"$i"
    cp tests/fixtures/default_aims_calcs/"$i"/*.in tests/fixtures/custom_bin_aims_calcs/"$i"

    # Run the binary in each of the directories
    cd tests/fixtures/custom_bin_aims_calcs/"$i" || exit 1
    mpirun -n 4 "$binary" >aims.out
); done
