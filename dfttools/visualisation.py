import matplotlib.pyplot as plt
import numpy as np

from dfttools.output import AimsOutput


class AimsVisualisation(AimsOutput):
    """FHI-aims visualisation tools.

    ...

    Attributes
    ----------
    scf_conv_acc_params : dict
        The SCF convergence accuracy parameters.
    """

    def plot_aims_convergence(
        self,
        scf_conv_acc_params=None,
        title=None,
        forces=False,
        ks_eigenvalues=False,
        fig_size=(18, 6),
        p2_y_scale="log",
        p3_ylim=(-500, 500),
    ) -> plt.figure.Figure:
        """Plot the SCF convergence accuracy parameters.

        Parameters
        ----------
        scf_conv_acc_params : dict
            The SCF convergence accuracy parameters.
        title : str
            System name to use in title of the plot.
        forces : bool
            Whether to plot the change of forces and forces on atoms.
        ks_eigenvalues : bool
            Whether to plot the Kohn-Sham eigenvalues.
        fig_size : tuple
            The total size of the figure.
        p2_y_scale : str
            The y-scale of the change of charge plot.
        p3_ylim : tuple
            The y-limits of the change of total energies/sum of eigenvalues plot.

        Returns
        -------
        plt.figure.Figure
            The matplotlib figure object.
        """

        # Get the SCF convergence accuracy parameters if not provided
        if scf_conv_acc_params is None:
            if not hasattr(self, "scf_conv_acc_params"):
                self.scf_conv_acc_params = self.get_i_scf_conv_acc()

        # Override the default scf_conv_acc_params if given in function
        else:
            self.scf_conv_acc_params = scf_conv_acc_params

        scf_iters = self.scf_conv_acc_params["scf_iter"]
        tot_scf_iters = np.arange(1, len(scf_iters) + 1)
        delta_charge = self.scf_conv_acc_params["change_of_charge"]
        delta_charge_sd = self.scf_conv_acc_params["change_of_charge_spin_density"]
        delta_sum_eigenvalues = self.scf_conv_acc_params["change_of_sum_eigenvalues"]
        delta_total_energies = self.scf_conv_acc_params["change_of_total_energy"]

        # Change the number of subplots if forces and ks_eigenvalues are to be plotted
        subplots = [True, True, forces, ks_eigenvalues]
        i_subplot = 1

        # Setup the figure subplots
        fig, ax = plt.subplots(1, subplots.count(True), figsize=fig_size)

        # Plot the change of charge
        ax[0].plot(tot_scf_iters, delta_charge, label=r"$\Delta$ charge")

        # Only plot delta_charge_sd if the calculation is spin polarised
        if delta_charge_sd is not None:
            ax[0].plot(
                tot_scf_iters, delta_charge_sd, label=r"$\Delta$ charge/spin density"
            )

        ax[0].set_yscale(p2_y_scale)
        ax[0].set_xlabel("SCF iter")
        ax[0].legend()
        if title is not None:
            ax[0].set_title(f"{title} change of charge")

        # Plot the change of total energies and sum of eigenvalues
        ax[1].plot(
            tot_scf_iters, delta_total_energies, label=r"$\Delta$ total energies"
        )
        ax[1].plot(
            tot_scf_iters,
            delta_sum_eigenvalues,
            label=r"$\Delta \; \Sigma$ eigenvalues",
        )
        ax[1].set_xlabel("SCF iter")
        ax[1].set_ylim(p3_ylim)
        ax[1].legend()
        if title is not None:
            ax[1].set_title(rf"{title} change of $\Sigma$ eigenvalues and total E")

        # Plot the forces
        if forces:
            i_subplot += 1
            delta_forces = self.scf_conv_acc_params["change_of_forces"]
            forces_on_atoms = self.scf_conv_acc_params["forces_on_atoms"]
            ax[i_subplot].plot(tot_scf_iters, delta_forces, label=r"$\Delta$ forces")
            ax[i_subplot].plot(tot_scf_iters, forces_on_atoms, label="Forces on atoms")
            ax[i_subplot].set_xlabel("SCF iter")
            ax[i_subplot].set_ylabel("Force (eV/Angstrom)")
            ax[i_subplot].legend()

            if title is not None:
                ax[i_subplot].set_title(f"{title} change of forces")

        # Plot the KS state energies
        if ks_eigenvalues:
            i_subplot += 1
            ks_eigenvals = self.get_all_ks_eigenvalues()

            if isinstance(ks_eigenvals, dict):
                for i, j in zip(ks_eigenvals["eigenvalue_eV"], ks_eigenvals["state"]):
                    ax[i_subplot].plot(tot_scf_iters, i, label=f"KS state {j}")

            elif isinstance(ks_eigenvals, tuple):
                su_ks_eigenvals = ks_eigenvals[0]
                sd_ks_eigenvals = ks_eigenvals[1]

                print(tot_scf_iters)
                print(su_ks_eigenvals)

                for i, j in zip(
                    su_ks_eigenvals["eigenvalue_eV"], su_ks_eigenvals["state"]
                ):
                    ax[i_subplot].plot(tot_scf_iters, i, label=f"Spin-up KS state {j}")

                for i, j in zip(
                    sd_ks_eigenvals["eigenvalue_eV"], sd_ks_eigenvals["state"]
                ):
                    ax[i_subplot].plot(
                        tot_scf_iters, i, label=f"Spin-down KS state {j}"
                    )

            ax[i_subplot].set_xlabel("SCF iter")
            ax[i_subplot].set_ylabel("Energy (eV)")
            # ax[i_subplot].legend()

            if title is not None:
                ax[i_subplot].set_title(f"{title} KS state energies")

        return fig
