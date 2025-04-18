from allatom_design.data.filter.dynamic.filter import DynamicFilter

from allatom_design.data.types import Record


class SelfConsistencyFilter(DynamicFilter):
    """A filter that filters structures based on their size."""

    def __init__(self, max_scrmsd: float) -> None:
        """Initialize the filter.

        Parameters
        ----------
        max_scrmsd : float
            The maximum scRMSD allowed.

        """
        self.max_scrmsd = max_scrmsd

    def filter(self, record: Record) -> bool:
        """Filter structures based on their self-consistency.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        return record.designability_info.sc_ca_rmsd <= self.max_scrmsd
