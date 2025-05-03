from collections import defaultdict

from allatom_design.data import const
from allatom_design.data.filter.dynamic.filter import DynamicFilter
from allatom_design.data.types import Record


class ChainTypeSizeFilter(DynamicFilter):
    """
    A filter for structures based on the number of chains of a given type and number of residues across all chains of that type.
    """

    def __init__(self,
                 chain_type: str,
                 min_chains: int | None,
                 max_chains: int | None,
                 min_residues: int | None,
                 max_residues: int | None) -> None:
        """
        All ranges are inclusive.

        min_residues and max_residues
        """
        if chain_type not in const.chain_types:
            raise ValueError(f"Invalid chain type for ChainTypesFilter: {chain_type}")
        self.chain_type = chain_type

        self.min_chains = min_chains
        self.max_chains = max_chains
        self.min_residues = min_residues
        self.max_residues = max_residues


    def filter(self, record: Record) -> bool:
        """Filter structures based on the number of chains of a given type and number of residues across all chains of that type.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        type_chains = [chain for chain in record.chains if const.chain_types[chain.mol_type] == self.chain_type and chain.valid]

        # Filter by number of chains of the given type
        n_chains = len(type_chains)
        if self.min_chains is not None and n_chains < self.min_chains:
            return False
        if self.max_chains is not None and n_chains > self.max_chains:
            return False

        # Filter by number of residues across all chains of the given type
        n_residues = sum([chain.num_residues for chain in type_chains])
        if self.min_residues is not None and n_residues < self.min_residues:
            return False
        if self.max_residues is not None and n_residues > self.max_residues:
            return False

        return True
