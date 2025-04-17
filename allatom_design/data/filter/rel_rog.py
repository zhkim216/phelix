from boltz.data.filter.dynamic.filter import DynamicFilter

from allatom_design.data.types import Record


class RelRogFilter(DynamicFilter):
    """A filter that filters structures based on their relative radius of gyration."""

    def __init__(self, max_rel_rog: float) -> None:
        """Initialize the filter.

        Parameters
        ----------
        max_rel_rog : float
            The maximum relative radius of gyration allowed.

        """
        self.max_rel_rog = max_rel_rog

    def filter(self, record: Record) -> bool:
        """Filter structures based on their relative radius of gyration.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        return record.designability_info.rel_rog <= self.max_rel_rog
