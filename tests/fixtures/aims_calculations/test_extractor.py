#!/usr/bin/env python3

import yaml
import glob

from dfttools.output import AimsOutput

aims_out_files = glob.glob("*/aims.out")


for file in aims_out_files:

    ao = AimsOutput(aims_out=file)

    # print(ao.check_exit_normal())
    # print(ao.check_spin_polarised())
    # print(ao.get_conv_params())
    # print(ao.get_final_energy())
    # print(ao.get_n_relaxation_steps())
    # print(ao.get_n_scf_iters())
    # print(ao.get_i_scf_conv_acc())
    # print(ao.get_n_initial_ks_states())
    # print(ao.get_all_ks_eigenvalues())
    # print(ao.get_final_ks_eigenvalues())
    # print(ao.get_pert_soc_ks_eigenvalues())

    # Write to yaml file
    i_scf_conv_acc = ao.get_i_scf_conv_acc()
    print(i_scf_conv_acc)
    with open("i_scf_conv_acc.yml", "w") as f:
        yaml.dump(i_scf_conv_acc.values(), f, default_flow_style=False)

    break
