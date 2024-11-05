import numpy as np
import numpy.typing as npt
from dfttoolkit.output import AimsOutput
import dfttoolkit.utils.file_utils as fu

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

    def get_timings_per_benchmark(self) -> List[npt.NDArray[np.float64]]:
        """
        Calculate the average time taken per SCF iteration for each benchmark.

        Returns
        -------
        List[np.ndarray]
            The average time taken per SCF iteration for each benchmark.
        """

        benchmark_timings = []

        for aims_out in self.benchmarks:
            scf_iter_times = aims_out.get_time_per_scf()
            benchmark_timings.append(scf_iter_times)

        return benchmark_timings
