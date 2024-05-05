import numpy as np
from dfttools.output import AimsOutput
import dfttools.utils.file_utils as fu

from typing import List


class BenchmarkAims(AimsOutput):
    """
    Calculate benchmarking metrics for FHI-aims calculations.

    ...

    Attributes
    ----------
    benchmark_dirs : List[str]
        The paths to the aims.out files.
    """

    def __init__(self, benchmark_dirs: List[str]):

        self.benchmarks = []

        # Get the aims.out files from the provided directories
        for aims_out in benchmark_dirs:
            ao = AimsOutput(aims_out=aims_out)
            self.benchmarks.append(ao)

    @staticmethod
    def get_time_per_scf(aims_out) -> np.ndarray:
        """
        Calculate the average time taken per SCF iteration.

        Parameters
        ----------
        aims_out : List[str]
            The aims.out file to parse.

        Returns
        -------
        np.ndarray
            The average time taken per SCF iteration.
        """

        # Get the number of SCF iterations
        n_scf_iters = aims_out.get_n_scf_iters()
        scf_iter_times = np.zeros(n_scf_iters)

        # Get the time taken for each SCF iteration
        iter_num = 0
        for line in aims_out.lines:
            if "Time for this iteration" in line:
                scf_iter_times[iter_num] = float(line.split()[-4])
                iter_num += 1

        return scf_iter_times

    def get_timings_per_benchmark(self) -> List[np.ndarray]:
        """
        Calculate the average time taken per SCF iteration for each benchmark.

        Returns
        -------
        List[np.ndarray]
            The average time taken per SCF iteration for each benchmark.
        """

        benchmark_timings = []

        for aims_out in self.benchmarks:
            scf_iter_times = self.get_time_per_scf(aims_out)
            benchmark_timings.append(scf_iter_times)

        return benchmark_timings
