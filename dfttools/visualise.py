import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes, figure
from matplotlib.ticker import MaxNLocator

from dfttools.output import AimsOutput


class VisualiseAims(AimsOutput):
    """
    FHI-aims visualisation tools.

    ...

    Attributes
    ----------
    scf_conv_acc_params : dict
        the SCF convergence accuracy parameters

    Methods
    -------
    convergence(scf_conv_acc_params=None, title=None, forces=False, ks_eigenvalues=False, fig_size=(24, 6))
        Plot the SCF convergence accuracy parameters.
    """

    @staticmethod
    def _plot_charge_convergence(
        ax,
        tot_scf_iters,
        delta_charge,
        delta_charge_sd=None,
        conv_params=None,
        title=None,
    ) -> axes.Axes:
        """
        Create a subplot for the charge convergence of an FHI-aims calculation.

        Parameters
        ----------
        ax : axes.Axes
            matplotlib subplot index
        tot_scf_iters : Union[numpy.ndarray, list]
            cumulative SCF iterations
        delta_charge : Union[numpy.ndarray, list]
            change of spin-up or total spin (if the calculation was spin none)
            eigenvalues
        delta_charge_sd : Union[Union[numpy.ndarray, list], None]
            change of spin-down eigenvalues
        conv_params : dict
            convergence parameters which determine if the SCF cycle has converged
        title : str
            system name to include in title

        Returns
        -------
        axes.Axes
            matplotlib subplot object
        """

        ax.plot(tot_scf_iters, delta_charge, label=r"$\Delta$ charge")

        # Only plot delta_charge_sd if the calculation is spin polarised
        if delta_charge_sd is not None:
            ax.plot(
                tot_scf_iters, delta_charge_sd, label=r"$\Delta$ charge/spin density"
            )

        # Add the convergence parameters
        if conv_params is not None and conv_params["charge_density"] != 0.0:
            ax.axhline(
                conv_params["charge_density"],
                ls="--",
                c="gray",
                label=r"$\rho$ convergence criterion",
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        ax.set_xlabel("cumulative SCF iteration")
        ax.set_ylabel(r"Charge / $e a_0^{-3}$")
        ax.legend()
        if title is not None:
            ax.set_title(rf"{title} $\Delta$ charge Convergence")

        return ax

    @staticmethod
    def _plot_energy_convergence(
        ax,
        tot_scf_iters,
        delta_sum_eigenvalues,
        delta_total_energies,
        conv_params=None,
        absolute=False,
        title=None,
    ) -> axes.Axes:
        """
        Create a subplot for the energy convergence of an FHI-aims calculation.

        Parameters
        ----------
        ax : axes.Axes
            matplotlib subplot index
        delta_sum_eigenvalues : Union[numpy.ndarray, list]
            change of sum of eigenvalues
        delta_total_energies : Union[numpy.ndarray, list]
            change of total energies
        conv_params : dict
            convergence parameters which determine if the SCF cycle has converged
        title : str
            system name to include in title

        Returns
        -------
        axes.Axes
            matplotlib subplot object
        """

        if absolute:
            delta_sum_eigenvalues = abs(delta_sum_eigenvalues)
            delta_total_energies = abs(delta_total_energies)

        ax.plot(
            tot_scf_iters,
            delta_sum_eigenvalues,
            label=r"$\Delta \; \Sigma$ eigenvalues",
            c="C1",
        )
        ax.plot(
            tot_scf_iters,
            delta_total_energies,
            label=r"$\Delta$ total energies",
            c="C0",
        )

        # Add the convergence parameters
        if conv_params is not None:
            ax.axhline(
                conv_params["sum_eigenvalues"],
                ls="-.",
                c="darkgray",
                label=r"$\Delta \; \Sigma$ eigenvalues convergence criterion",
            )
            ax.axhline(
                conv_params["total_energy"],
                ls="--",
                c="gray",
                label=r"$\Delta$ total energies convergence criterion",
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        ax.set_xlabel("cumulative SCF iteration")
        ax.set_ylabel(r"absolute energy / $| \mathrm{eV} |$")
        ax.legend()
        if title is not None:
            ax.set_title(rf"{title} Energies and Eigenvalues Convergence")

        return ax

    @staticmethod
    def _plot_forces_convergence(ax, forces_on_atoms, conv_params=None, title=None):
        """
        Create a subplot for the forces convergence of an FHI-aims calculation.

        Parameters
        ----------
        ax : axes.Axes
            matplotlib subplot index
        delta_forces : Union[numpy.ndarray, list]
            change of forces
        forces_on_atoms : Union[numpy.ndarray, list]
            all forces acting on each atom
        conv_params : dict
            convergence parameters which determine if the SCF cycle has converged
        title : str
            system name to include in title
        """

        # see NOTE in dfttools.output.AimsOutput.get_i_scf_conv_acc()
        # ax.plot(delta_forces, label=r"$\Delta$ forces")
        ax.plot(forces_on_atoms, label="forces on atoms")

        # Add the convergence parameters
        if conv_params is not None and conv_params["total_force"] is not None:
            ax.axhline(
                conv_params["total_force"],
                ls="--",
                c="gray",
                label="forces convergence criterion",
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("geometry relaxation step")
        ax.set_ylabel(r"force / $\mathrm{eV} \mathrm{\AA}^{-1}$")
        ax.legend()

        if title is not None:
            ax.set_title(rf"{title} Forces Convergence")

    @staticmethod
    def _plot_ks_states_convergence(ax, ks_eigenvals, title=None):
        """
        Create a subplot for the energy changes of the Kohn-Sham eigenstates in an FHI-aims calculation.

        Parameters
        ----------
        ax : axes.Axes
            matplotlib subplot index
        ks_eigenvals : Union[dict, Tuple[numpy.ndarray, numpy.ndarray]]
            state, occupation, and eigenvalue of each KS state at each SCF
            iteration
        title : str
            system name to include in title

        Returns
        -------
        axes.Axes
            matplotlib subplot object
        """

        if isinstance(ks_eigenvals, dict):
            # Don't include last eigenvalue as it only prints after final SCF iteration
            # Add 1 to total SCF iterations to match the length of the eigenvalues and
            # we want to include the first pre SCF iteration
            for ev in ks_eigenvals["eigenvalue_eV"][:-2].T:
                ax.plot(np.arange(1, len(ks_eigenvals["eigenvalue_eV"]) - 1), ev)

        elif isinstance(ks_eigenvals, tuple):
            su_ks_eigenvals = ks_eigenvals[0]
            sd_ks_eigenvals = ks_eigenvals[1]

            for ev in su_ks_eigenvals["eigenvalue_eV"][:-2].T:
                ax.plot(
                    np.arange(1, len(su_ks_eigenvals["eigenvalue_eV"]) - 1), ev, c="C0"
                )

            for ev in sd_ks_eigenvals["eigenvalue_eV"][:-2].T:
                ax.plot(
                    np.arange(1, len(su_ks_eigenvals["eigenvalue_eV"]) - 1), ev, c="C1"
                )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("symlog")
        ax.set_xlabel("cumulative SCF iteration")
        ax.set_ylabel("energy / eV")

        if title is not None:
            ax.set_title(f"{title} KS State Convergence")

    def convergence(
        self,
        conv_params=None,
        scf_conv_acc_params=None,
        title=None,
        forces=False,
        ks_eigenvalues=False,
        fig_size=(24, 6),
    ) -> figure.Figure:
        """
        Plot the SCF convergence accuracy parameters.

        Parameters
        ----------
        conv_params: dict
            convergence parameters which determine if the SCF cycle has converged
        scf_conv_acc_params : dict
            the scf convergence accuracy parameters
        title : str
            system name to use in title of the plot
        forces : bool
            whether to plot the change of forces and forces on atoms
        ks_eigenvalues : bool
            whether to plot the kohn-sham eigenvalues
        fig_size : tuple
            the total size of the figure

        Returns
        -------
        figure.Figure
            matplotlib figure object
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
        self._plot_charge_convergence(
            ax[0], tot_scf_iters, delta_charge, delta_charge_sd, conv_params, title
        )

        # Plot the change of total energies and sum of eigenvalues
        self._plot_energy_convergence(
            ax[1],
            tot_scf_iters,
            delta_sum_eigenvalues,
            delta_total_energies,
            conv_params,
            True,
            title,
        )

        # Plot the forces
        if forces:
            # see NOTE in dfttools.output.AimsOutput.get_i_scf_conv_acc()
            # delta_forces = self.scf_conv_acc_params["change_of_forces"]
            # delta_forces = np.delete(delta_forces, np.argwhere(delta_forces == 0.0))
            forces_on_atoms = self.scf_conv_acc_params["forces_on_atoms"]
            forces_on_atoms = np.delete(
                forces_on_atoms, np.argwhere(forces_on_atoms == 0.0)
            )
            i_subplot += 1
            self._plot_forces_convergence(
                ax[i_subplot], forces_on_atoms, conv_params, title
            )

        # Plot the KS state energies
        if ks_eigenvalues:
            i_subplot += 1
            ks_eigenvals = self.get_all_ks_eigenvalues()

            self._plot_ks_states_convergence(ax[i_subplot], ks_eigenvals, title)

        return fig
