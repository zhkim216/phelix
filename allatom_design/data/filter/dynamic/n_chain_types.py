from collections import defaultdict

from allatom_design.data import const
from allatom_design.data.filter.dynamic.filter import DynamicFilter
from allatom_design.data.types import Record


class NChainTypesFilter(DynamicFilter):
    """A filter that filters structures based on the number of chains of each type."""

    def __init__(self,
                 n_protein_chain_range: tuple[int, int],
                 n_dna_chain_range: tuple[int, int],
                 n_rna_chain_range: tuple[int, int],
                 n_nonpolymer_chain_range: tuple[int, int]) -> None:
        """
        All ranges are inclusive.
        """
        # self.n_protein_chain_range = n_protein_chain_range
        # self.n_dna_chain_range = n_dna_chain_range
        # self.n_rna_chain_range = n_rna_chain_range
        # self.n_nonpolymer_chain_range = n_nonpolymer_chain_range
        self.chain_type_ranges = {
            "PROTEIN": n_protein_chain_range,
            "DNA": n_dna_chain_range,
            "RNA": n_rna_chain_range,
            "NONPOLYMER": n_nonpolymer_chain_range,
        }


    def filter(self, record: Record) -> bool:
        """Filter structures based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        chain_counts = self._count_chain_types(record)
        for chain_type, count in chain_counts.items():
            min_n, max_n = self.chain_type_ranges[chain_type]
            if count < min_n or count > max_n:
                return False
        return True


    def _count_chain_types(self, record: Record) -> dict[str, int]:
        """Count the number of valid chains of each type."""
        chain_counts = defaultdict(int)
        for chain in record.chains:
            if not chain.valid:
                continue
            chain_type = const.chain_types[chain.mol_type]
            chain_counts[chain_type] += 1
        return chain_counts
