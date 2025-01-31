#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, isfile, dirname, basename, isdir
import os

from dfttoolkit.output import AimsOutput
import numpy as np
import argparse
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def get_all_outputs(calc_dir, outputname):
    restart_dirs = [join(calc_dir, d) for d in os.listdir(calc_dir)]
    restart_dirs = sorted(
        [d for d in restart_dirs if isfile(join(d, outputname))]
    )
    restart_dirs.append(calc_dir)

    outputs = []
    for d in restart_dirs:
        outputs.append(AimsOutput(join(d, outputname)))

    return outputs


def get_energies(aims_outputs):
    E = []
    for aims in aims_outputs:
        E += list(aims.get_energy_corrected(None))
    return np.array(E)


def get_maximum_forces(aims_outputs):
    force = []
    for aims in aims_outputs:
        force += list(aims.get_maximum_force(None))
        # force += list(aims.get_remaining_force(None))
    return np.array(force)


def get_geometries(aims_outputs):
    geometries = []
    for aims in aims_outputs:
        geometries += aims.get_geometry_steps_of_optimisation()
    return geometries


def plot_energy(E, logarithmic=False):
    E_ref = np.min([e for e in E if not np.isnan(e)])
    E_difference = 1e3 * (E - E_ref)
    print(E, np.min([e for e in E if not np.isnan(e)]))

    ymax_rounded_to_next_power_of_ten = 10 ** np.ceil(
        np.log10(E_difference.max())
    )

    if logarithmic:
        # reference energy would be zero per def. -> plot np.NAN instead
        E_difference_replace = E_difference.copy()
        E_difference_replace[E_difference_replace == 0.0] = np.nan
        plt.semilogy(
            range(len(E)), E_difference_replace, marker="+", linewidth=2
        )
        plt.ylim(
            [
                0.9 * sorted(E_difference)[1],  # use lowest plotted E_diff
                ymax_rounded_to_next_power_of_ten,
            ]
        )
    else:
        plt.plot(range(len(E)), E_difference, marker="+", linewidth=2)
        plt.ylim([0, ymax_rounded_to_next_power_of_ten])

    plt.ylabel("$\Delta$ E / meV")
    plt.xlim([0, len(E)])


def plot_forces(forces, converged, threshold, logarithmic=False):
    ax2 = plt.subplot(1, 2, 2)
    if converged:
        color_key = "g"
    else:
        color_key = "r"

    if logarithmic:
        plt.semilogy(
            range(len(forces)), forces, color_key, marker="+", linewidth=2
        )
        ymax_rounded_to_next_power_of_ten = 10 ** np.ceil(
            np.log10(forces.max())
        )
        ymin_rounded_to_next_power_of_ten = 10 ** np.floor(
            np.log10(forces.min())
        )
        plt.ylim(
            [
                ymin_rounded_to_next_power_of_ten,
                ymax_rounded_to_next_power_of_ten,
            ]
        )
    else:
        plt.plot(
            range(len(forces)), forces, color_key, marker="+", linewidth=2
        )
        plt.ylim([0, 1.0])

    plt.ylabel("force / eV/A")
    if threshold is None:
        plt.axhline(0.15, color="k", linestyle="--")
        plt.axhline(0.10, color="k", linestyle="--")
        plt.axhline(0.05, color="k", linestyle="--")
    else:
        plt.axhline(
            threshold,
            color="k",
            linestyle="--",
            label="threshold = {} eV/A".format(threshold),
        )

    plt.axhline(
        forces[-1],
        color=color_key,
        linestyle="--",
        label="last value = {:.3} eV/A".format(forces[-1]),
    )
    # plt.text(x=len(forces)/2, y=forces[-1] + 0.01, s="{:.3} eV/A".format(forces[-1]))

    plt.xlim([0, len(forces)])
    ax2.legend()


def plot_energy_and_forces(
    E,
    forces,
    converged,
    threshold,
    plot_energy_logarithmic=False,
    plot_forces_logarithmic=False,
):
    plt.close("all")
    fig = plt.figure()
    plt.subplot(1, 2, 1)

    plot_energy(E, logarithmic=plot_energy_logarithmic)
    plot_forces(forces, converged, threshold, plot_forces_logarithmic)

    fig.tight_layout()


def plot_height(h):
    plt.close("all")
    plt.figure()
    plt.ylabel("Distance C-slab / A")
    plt.plot(range(len(h)), h, linewidth=2)


def get_heights(geometries):
    """Get height of uppermost slab layer and C atoms"""
    heights = []
    try:
        N_layers = int(input("Number of slab layers:"))
    except:
        Exception("Input must be a string!")

    for g in geometries:
        spec = g.species
        species = [s[:2] for s in spec]
        slab = max(set(species), key=species.count)

        C_height = 0
        slab_heights = []
        count = 0
        for i, s in enumerate(species):
            if "C" in s:
                C_height += g.coords[i, 2]
                count += 1

            if slab in s:
                slab_heights.append(g.coords[i, 2])

        upper_layer = sorted(slab_heights)[-N_layers:]
        slab_height = np.mean(np.array(upper_layer))

        if not count:
            C_height = None
            Warning("No C atoms found")
        else:
            C_height /= count
        heights.append(C_height - slab_height)

    return heights


def get_output_filename(
    output_option, calc_dir, energy_logarithmic, forces_logarithmic
):

    file_name = "optimization"
    if energy_logarithmic:
        file_name += "_log-energy"
    if forces_logarithmic:
        file_name += "_log-forces"
    file_name += ".png"

    if output_option == "":
        return join(calc_dir, file_name)

    if isdir(output_option):
        return join(output_option, basename(calc_dir))

    if output_option[-4:].lower() in [".png", ".pdf", ".svg"]:
        return output_option
    else:
        raise NotImplementedError(
            "Invalid figure filename or directory does not exist."
        )


def visualise_optimisation(aims_output, args):
    outputname = basename(aims_output)
    calc_dir = dirname(aims_output)
    if calc_dir == "":
        calc_dir = "."

    # read geometries and forces
    aims_outputs = get_all_outputs(calc_dir, outputname)
    try:
        control_file = aims_outputs[-1].getControlFile()
        trm_text = control_file.settings["relax_geometry"]
        threshold = np.float(trm_text.split()[1])
    except:
        threshold = None
    E = get_energies(aims_outputs)
    force = get_maximum_forces(aims_outputs)
    converged = aims_outputs[-1].check_geometry_optimisation_has_completed()

    # Create and save figure
    if args.both_logarithmic:
        plot_energy_logarithmic = True
        plot_forces_logarithmic = True
        postfix = ""
    else:
        plot_energy_logarithmic = args.energy_logarithmic
        plot_forces_logarithmic = args.forces_logarithmic
    plot_energy_and_forces(
        E,
        force,
        converged,
        threshold,
        plot_energy_logarithmic=plot_energy_logarithmic,
        plot_forces_logarithmic=plot_forces_logarithmic,
    )
    outputname = get_output_filename(
        args.output, calc_dir, plot_energy_logarithmic, plot_forces_logarithmic
    )
    plt.savefig(outputname, dpi=600)

    geometries = None
    # Save all geometry steps of the optimization
    if args.geometries:
        geometries_dir = join(calc_dir, "geometries")
        if not isdir(geometries_dir):
            os.makedirs(geometries_dir)
        geometries = get_geometries(aims_outputs)
        for i, g in enumerate(geometries):
            # try:
            fig = plt.Figure()
            g.visualise()
            fig.savefig(join(geometries_dir, "{:04d}.png".format(i)), dpi=600)

            # except:
            #    print("Print with ASE failed")
            g.save_to_file(join(geometries_dir, "{:04d}.in".format(i)))

        if args.gif:
            import imageio, glob

            with imageio.get_writer("optimization.gif") as writer:
                for g in sorted(glob.glob(join(geometries_dir, "*.png"))):
                    writer.append_data(imageio.imread(g))

    if args.displacement:
        if geometries is None:
            geometries = get_geometries(aims_outputs)
        try:
            plt.figure()
            geometries[0].visualize()
            geometries[0].visualizeAtomDisplacements(geometries[-1])
            plt.savefig(join(calc_dir, "atom_displacements.png"), dpi=600)
        # TODO: find out which exceptions could be thrown
        except:
            print("visualizing atom displacements failed")

    # Save forces of all geometry steps of the optimisation
    if args.forces:
        forces_dir = join(calc_dir, "forces")
        if not isdir(forces_dir):
            os.makedirs(forces_dir)
        ind = 0
        geometries = get_geometries(aims_outputs)
        # print(aims_outputs, len(aims_outputs))
        # print(geometries, len(geometries))
        for i, ao in enumerate(aims_outputs):  # , geometries)):
            try:
                for j in range(len(ao.getEnergyCorrected(None))):
                    F = ao.getGradients(nr_of_occurrence=j)
                    geom = geometries[ind]

                    plt.close()
                    fig = plt.figure()
                    geom.visualize()
                    # ao.visualizeForces(nr_of_occurrence = j)
                    geom.visualizeForces(
                        F,
                        axes=[0, 1],
                        arrow_scale=20,
                        print_constrained=False,
                    )
                    fig.savefig(
                        join(forces_dir, "{:04d}.png".format(ind)), dpi=600
                    )
                    plt.close()
                    ind = ind + 1

            except:
                print("Printing forces failed")
                raise
    # Create height plot
    if args.heightaboveslab:
        geometries = get_geometries(aims_outputs)
        heights = get_heights(geometries)
        plot_height(heights)
        plt.savefig(outputname[:-4] + "_height" + outputname[-4:], dpi=600)


if __name__ == "__main__":
    usage_example = """
    Usage example:
    visualizeAIMSOptimization.py myCalc/aims.out
    visualizeAIMSOptimization.py */aims.out -o path/to/figures/directory
    """
    parser = argparse.ArgumentParser(
        description="Visualize energy and forces during a geometry optimization.",
        epilog=usage_example,
    )
    parser.add_argument(
        "aims_output", nargs="+", help="Path to aims output file(s)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Output directory or name for figure. Set to empty directory to save in the calculation directory.",
    )
    parser.add_argument(
        "--geometries",
        "-g",
        action="store_true",
        help="Dump images of all geometry steps of the optimization.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Creates a gif from the images of the geometry steps. Requires --geometries.",
    )
    parser.add_argument(
        "--displacement",
        "-d",
        action="store_true",
        help="Visualize atom displacements of geometry optimization",
    )
    parser.add_argument(
        "--heightaboveslab",
        "-s",
        action="store_true",
        help="Add plot of height difference between C and slab",
    )
    parser.add_argument(
        "--forces",
        "-f",
        action="store_true",
        help="Add plot of forces for each geometry step",
    )
    parser.add_argument(
        "--logarithmic",
        "-l",
        action="store_true",
        default=False,
        dest="both_logarithmic",
        help='Plot both energy differences and maximum force logarithmically. Shorthand for option "-el -fl".',
    )
    parser.add_argument(
        "--energy-logarithmic",
        "-el",
        action="store_true",
        default=False,
        dest="energy_logarithmic",
        help="Plot log(Delta E) instead of Delta E.",
    )
    parser.add_argument(
        "--forces-logarithmic",
        "-fl",
        action="store_true",
        default=False,
        dest="forces_logarithmic",
        help="Plot log(max_force) instead of max_force.",
    )

    #    args = ["/home/michael/calculations/TCNE_Ag/Ag4/Batch1/(5, 0, 0, 6, 3, 3, 5, 5, 0, 15, 12, 27)/aims.out"]
    args = parser.parse_args()

    # manage dependencies of arguments
    if args.gif and not args.geometries:
        parser.error("--gif requires the arguments --geometries or -g")

    for aims in args.aims_output:
        visualise_optimisation(aims, args)
