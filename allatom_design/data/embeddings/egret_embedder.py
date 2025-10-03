from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd


class MissingDependencyError(ImportError):
    """Raised when a required dependency is not available at runtime."""


class EgretEmbedder:
    """Compute atom-level embeddings from Egret-1 family (MACE) models.

    This class wraps the mace-torch "mace_off" calculator to obtain per-atom
    node features ("node_feats"). It also supports converting the full
    equivariant descriptor to invariant descriptors according to guidance from
    the Egret public repository.

    References:
      - Egret public repo: https://github.com/rowansci/egret-public
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        default_dtype: Literal["float32", "float64"] = "float64",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> None:
        """Initialize the Egret embedder.

        Args:
            model_path: Filesystem path to an Egret-1 family model file
                (e.g., "EGRET_1.model").
            default_dtype: Default floating point precision used by the
                calculator. Egret examples commonly use "float64".
            device: Torch device to run on. "auto" selects CUDA when
                available, otherwise CPU.
        """
        self.model_path = str(model_path)
        self.default_dtype = str(default_dtype)
        self.device: str = self._resolve_device(device)

        # Lazy imports to minimize environment surface area if unused
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise MissingDependencyError(
                "PyTorch is required to run Egret embeddings. Please ensure torch is installed in your environment."
            ) from exc

        try:
            from mace.calculators import mace_off  # type: ignore
        except Exception as exc:
            raise MissingDependencyError(
                "mace-torch is required. Install with: pip install mace-torch"
            ) from exc

        self._torch = __import__("torch")
        self._ase_io = None  # set on first use
        self._mace_off = mace_off

        # Build calculator
        # Note: mace_off constructs all needed internal modules
        self.calculator = self._mace_off(
            model=self.model_path,
            default_dtype=self.default_dtype,
        )

        # Move underlying model to the requested device if possible
        try:
            if self.device == "cuda":
                # The calculator holds models in a list
                for m in self.calculator.models:
                    m.to("cuda")
            # Ensure eval mode to avoid any training-time gradients unless requested
            for m in self.calculator.models:
                try:
                    m.eval()
                except Exception:
                    pass
        except Exception:
            # Do not fail hard on device transfer; CPU inference will still work
            pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Return a concrete device string from a user preference."""
        if device in ("cpu", "cuda"):
            return device
        try:
            import torch  # noqa: F401
            return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_ase(self) -> None:
        """Import ASE only when needed to reduce baseline dependencies."""
        if self._ase_io is None:
            try:
                import ase.io as ase_io  # type: ignore
            except Exception as exc:
                raise MissingDependencyError(
                    "ASE is required to read structure files. Install with: pip install ase"
                ) from exc
            self._ase_io = ase_io

    # --------------------------- Core computation --------------------------- #

    def load_atoms(self, structure_path: str | Path, *, frame_index: int = 0):
        """Load a structure file into an ASE Atoms object.

        Args:
            structure_path: Path to a molecular structure file (PDB/mmCIF/SDF/etc.).
            frame_index: For multi-frame files, the frame to read (0-based).

        Returns:
            ASE Atoms instance.
        """
        self._ensure_ase()
        structure_path = str(structure_path)
        atoms = self._ase_io.read(structure_path, index=frame_index)
        return atoms

    def _compute_full_node_feats(self, atoms) -> "np.ndarray":
        """Compute full equivariant node features (shape: [L, D]).

        This calls the underlying torch model directly as suggested in the
        Egret repo examples, using the calculator's private helper to build
        the batch dict.
        """
        torch = self._torch
        batch = self.calculator._atoms_to_batch(atoms).to_dict()
        # Avoid force computation failures by enabling gradients w.r.t positions
        # when the scripted model requests forces by default.
        try:
            if isinstance(batch.get("positions", None), torch.Tensor):
                batch["positions"] = batch["positions"].detach().requires_grad_(True)
        except Exception:
            pass
        outputs = self.calculator.models[0](batch)
        node_feats = outputs["node_feats"]  # Tensor[L, D]
        return node_feats.detach().cpu().numpy()

    @staticmethod
    def to_invariant_descriptor(full_descriptor: np.ndarray) -> np.ndarray:
        """Convert full (equivariant) descriptor to invariant descriptor.

        Rules per Egret docs:
          - Standard models: full D=1920 -> invariant is concat(first 192, last 192) => [L, 384]
          - Egret-1M: full D=640 -> invariant is concat(first 128, last 128) => [L, 256]
          - Egret-1S: full D=192 -> already invariant => [L, 192]

        Raises on unknown descriptor width to avoid producing misleading results.
        """
        if full_descriptor.ndim != 2:
            raise ValueError("Descriptor must be a 2D array of shape [L, D].")
        D = int(full_descriptor.shape[1])
        if D == 1920:
            return np.concatenate([full_descriptor[:, :192], full_descriptor[:, -192:]], axis=1)
        if D == 640:
            return np.concatenate([full_descriptor[:, :128], full_descriptor[:, -128:]], axis=1)
        if D == 192:
            return full_descriptor
        raise ValueError(
            f"Unsupported full descriptor width {D}. Supported: 1920 (Egret-1), 640 (Egret-1M), 192 (Egret-1S)."
        )

    def embed(
        self,
        structure_path: str | Path,
        *,
        frame_index: int = 0,
        descriptor: Literal["invariant", "equivariant"] = "invariant",
        return_dataframe: bool = True,
    ) -> dict:
        """Compute embeddings for a single structure and assemble outputs.

        Args:
            structure_path: PDB/mmCIF/SDF path.
            frame_index: Multi-frame index to read.
            descriptor: "invariant" or "equivariant".
            return_dataframe: If True, include a pandas DataFrame in results.

        Returns:
            Dict containing fields:
              - embedding: np.ndarray of shape [L, D]
              - atoms_count: int
              - structure_path: str
              - df (optional): pandas.DataFrame with atom metadata and embedding list column
        """
        atoms = self.load_atoms(structure_path, frame_index=frame_index)
        full = self._compute_full_node_feats(atoms)
        if descriptor == "invariant":
            emb = self.to_invariant_descriptor(full)
        elif descriptor == "equivariant":
            emb = full
        else:
            raise ValueError("descriptor must be 'invariant' or 'equivariant'")

        result: dict = {
            "embedding": emb,
            "atoms_count": int(emb.shape[0]),
            "structure_path": str(structure_path),
        }

        if return_dataframe:
            df = self._build_dataframe(atoms, emb, structure_path)
            result["df"] = df

        return result

    # ------------------------------ I/O helpers ------------------------------ #

    @staticmethod
    def _build_dataframe(atoms, embedding: np.ndarray, structure_path: str | Path) -> pd.DataFrame:
        """Assemble a wide DataFrame containing atom metadata and embeddings.

        The embedding is stored in a single list-typed column for Parquet
        compatibility (pyarrow list<item: double/float>). This avoids creating
        thousands of columns.
        """
        symbols = atoms.get_chemical_symbols()
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()  # [L, 3]

        rows = []
        for idx in range(len(atoms)):
            rows.append(
                {
                    "structure_path": str(structure_path),
                    "atom_index": int(idx),
                    "symbol": str(symbols[idx]),
                    "atomic_number": int(atomic_numbers[idx]),
                    "x": float(positions[idx, 0]),
                    "y": float(positions[idx, 1]),
                    "z": float(positions[idx, 2]),
                    "embedding": embedding[idx, :].astype(np.float32).tolist(),
                }
            )

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def _safe_to_parquet(df: pd.DataFrame, out_path: str | Path) -> bool:
        """Try writing a parquet file using available engines.

        Returns True on success, False otherwise.
        """
        out_path = str(out_path)
        # Try pyarrow first
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(out_path, index=False)
            return True
        except Exception:
            pass
        # Try fastparquet next
        try:
            import fastparquet  # noqa: F401
            df.to_parquet(out_path, engine="fastparquet", index=False)
            return True
        except Exception:
            return False

    @staticmethod
    def save_outputs(
        df: pd.DataFrame,
        embedding: np.ndarray,
        *,
        base_out_path: str | Path,
    ) -> str:
        """Save outputs to Parquet (preferred) or NPZ (fallback).

        Args:
            df: Atom table including the embedding list column.
            embedding: Raw embedding array [L, D] for NPZ fallback.
            base_out_path: Path without extension; ".parquet" or ".npz" will be used.

        Returns:
            The full path written.
        """
        base_out_path = Path(base_out_path)
        parquet_path = base_out_path.with_suffix(".parquet")
        if EgretEmbedder._safe_to_parquet(df, parquet_path):
            return str(parquet_path)

        # NPZ fallback (no external deps)
        npz_path = base_out_path.with_suffix(".npz")
        # Minimal metadata arrays
        atom_index = df["atom_index"].to_numpy()
        atomic_number = df["atomic_number"].to_numpy()
        symbol = df["symbol"].to_numpy()
        coords = df[["x", "y", "z"]].to_numpy()
        np.savez(
            str(npz_path),
            embedding=embedding,
            atom_index=atom_index,
            atomic_number=atomic_number,
            symbol=symbol,
            coords=coords,
        )
        return str(npz_path)

    # ------------------------------ Batch runner ----------------------------- #

    def embed_many(
        self,
        inputs: Sequence[str | Path],
        *,
        descriptor: Literal["invariant", "equivariant"] = "invariant",
        frame_index: int = 0,
        out_dir: Optional[str | Path] = None,
    ) -> list[str]:
        """Compute embeddings for many input files and write to disk.

        Args:
            inputs: Iterable of structure file paths.
            descriptor: Descriptor type to export.
            frame_index: Frame index for multi-frame files.
            out_dir: Output directory. If None, results are not saved.

        Returns:
            List of output file paths written (empty if out_dir is None).
        """
        written: list[str] = []
        out_dir_path = Path(out_dir) if out_dir is not None else None
        if out_dir_path is not None:
            out_dir_path.mkdir(parents=True, exist_ok=True)

        for p in inputs:
            res = self.embed(p, frame_index=frame_index, descriptor=descriptor, return_dataframe=True)
            if out_dir_path is None:
                continue
            base_name = Path(p).name
            base_no_ext = base_name
            # Keep original extension visible to avoid collisions between same basenames
            base_out = out_dir_path / f"{base_no_ext}"
            out_path = self.save_outputs(res["df"], res["embedding"], base_out_path=base_out)
            written.append(out_path)
        return written


