from allatom_design.data.types import Record
from allatom_design.data.filter.dynamic.filter import DynamicFilter


class ResolutionFilter(DynamicFilter):
    """A filter that filters complexes based on their resolution."""

    def __init__(self, resolution: float = 9.0) -> None:
        """Initialize the filter.

        Parameters
        ----------
        resolution : float, optional
            The maximum allowed resolution.

        """
        self.resolution = resolution

    def filter(self, record: Record) -> bool:
        """Filter complexes based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        structure = record.structure
        if structure.resolution == 0:
            # resolution of 0 means no resolution found in the PDB file
            return False

        return structure.resolution <= self.resolution
