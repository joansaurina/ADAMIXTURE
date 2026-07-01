import logging
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch

from .utils_c import (
    flip_packed,
    flip_unpacked,
    get_mean_packed,
    get_mean_unpacked,
    pack_genotypes,
    read_vcf_file,
    read_vcf_file_packed,
    replace_missing_with_three,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class SNPReader:
    """
    Wrapper to read genotype data from several formats.
    """

    def _parse_chromosome_number(self, chrom: str) -> int | None:
        """
        Description:
        Parses plain numeric chromosome labels and common chr-prefixed labels.

        Args:
            chrom (str): Chromosome label from the variant metadata.

        Returns:
            int | None: Parsed chromosome number, or None if the label is non-numeric.
        """
        chrom = chrom.strip()
        if chrom.lower().startswith("chr"):
            chrom = chrom[3:]
        if not chrom.isdigit():
            return None
        return int(chrom)

    def _keep_chromosome(self, chrom: str, chromosome_mode: str, autosome_count: int) -> bool:
        """
        Description:
        Decides whether a variant should be kept under the configured chromosome filter.

        Args:
            chrom (str): Chromosome label from the variant metadata.
            chromosome_mode (str): Chromosome filter mode ("all" or "autosomes").
            autosome_count (int): Number of autosomes kept when chromosome_mode is "autosomes".

        Returns:
            bool: True if the variant should be kept, otherwise False.
        """
        if chromosome_mode == "all":
            return True
        if chromosome_mode != "autosomes":
            raise ValueError("chromosome_mode must be 'all' or 'autosomes'")
        if autosome_count < 1:
            raise ValueError("autosome_count must be at least 1")

        chrom_num = self._parse_chromosome_number(chrom)
        return chrom_num is not None and 1 <= chrom_num <= autosome_count

    def _log_chromosome_filter(self, skipped: int, chromosome_mode: str, autosome_count: int) -> None:
        """
        Description:
        Logs a warning when variants are skipped by the chromosome filter.

        Args:
            skipped (int): Number of skipped variants.
            chromosome_mode (str): Chromosome filter mode ("all" or "autosomes").
            autosome_count (int): Number of autosomes kept when chromosome_mode is "autosomes".

        Returns:
            None
        """
        if skipped <= 0:
            return
        if chromosome_mode == "autosomes":
            log.warning(
                f"        Warning: Skipped {skipped} SNPs outside autosomes 1..{autosome_count}."
            )
        else:
            log.warning(f"        Warning: Skipped {skipped} SNPs excluded by chromosome filter.")

    def _get_base_path(self, file: str) -> str:
        """
        Description:
        Determines the base path by stripping known genotype extensions.

        Args:
            file (str): Input genotype file path.

        Returns:
            str: Base path without a known genotype extension.
        """
        file_str = str(file)
        for ext in ['.bed', '.vcf.gz', '.vcf', '.pgen', '.psam', '.pvar', '.fam', '.bim']:
            if file_str.endswith(ext):
                return file_str[:-len(ext)]
        return str(Path(file).with_suffix(''))

    def _read_bed(self, file: str, packed: bool, chunk_size: int, chromosome_mode: str, autosome_count: int) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Internal reader for PLINK BED files. Handles both regular (uint8) and
        packed (2-bit) formats for GPU acceleration.

        Args:
            file (str): Path to the BED file (without extension or with .bed).
            packed (bool): If True, returns a 2-bit packed torch.Tensor. Defaults to False.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        log.info("    Input format is BED.")

        base_path = self._get_base_path(file)
        fam_file = base_path + ".fam"
        bed_file = base_path + ".bed"

        with open(fam_file) as fam:
            N = sum(1 for _ in fam)
        N_bytes = ceil(N / 4)
        file_size = Path(bed_file).stat().st_size
        assert ((file_size - 3) % N_bytes) == 0, "bim file doesn't match!"
        M_total = (file_size - 3) // N_bytes

        bim_file = base_path + ".bim"
        keep_mask = []
        with open(bim_file) as bim:
            for line in bim:
                parts = line.strip().split()
                if not parts:
                    continue
                keep_mask.append(self._keep_chromosome(parts[0], chromosome_mode, autosome_count))
        keep_mask = np.array(keep_mask, dtype=bool)
        assert len(keep_mask) == M_total, "bim file doesn't match!"

        skipped = len(keep_mask) - keep_mask.sum()
        self._log_chromosome_filter(skipped, chromosome_mode, autosome_count)

        keep_idxs = np.flatnonzero(keep_mask).astype(np.uint32)
        M = keep_idxs.size

        if not packed:
            import pgenlib as pg

            G_raw = np.empty((M, N), dtype=np.int8)
            pgen_reader = pg.PgenReader(
                bed_file.encode(),
                raw_sample_ct=N,
                variant_ct=keep_mask.size,
            )
            try:
                if M > 0:
                    pgen_reader.read_list(keep_idxs, G_raw)
            finally:
                pgen_reader.close()

            replace_missing_with_three(G_raw)
            G = G_raw.view(np.uint8)
            return G, N, M

        log.info("        Reading BED in packed 2-bit format for GPU use.")
        import pgenlib as pg

        M_bytes = (M + 3) // 4
        G_packed = torch.zeros((M_bytes, N), dtype=torch.uint8)
        variants_per_chunk = max(4, (chunk_size // 4) * 4)
        G_chunk = np.empty((variants_per_chunk, N), dtype=np.int8)

        pgen_reader = pg.PgenReader(
            bed_file.encode(),
            raw_sample_ct=N,
            variant_ct=M_total,
        )
        try:
            for start in range(0, M, variants_per_chunk):
                stop = min(start + variants_per_chunk, M)
                chunk_len = stop - start
                packed_start = start // 4
                packed_len = (chunk_len + 3) // 4
                G_chunk_view = G_chunk[:chunk_len]

                pgen_reader.read_list(keep_idxs[start:stop], G_chunk_view)
                replace_missing_with_three(G_chunk_view)
                pack_genotypes(
                    G_chunk_view.view(np.uint8).ctypes.data,
                    G_packed[packed_start].data_ptr(),
                    chunk_len,
                    N,
                    packed_len,
                )
        finally:
            pgen_reader.close()

        return G_packed, N, M

    def _read_vcf(self, file: str, packed: bool, chunk_size: int, chromosome_mode: str, autosome_count: int) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Internal reader for VCF files using Cython-based parser.
        Handles both regular (uint8) and packed (2-bit) formats for GPU acceleration.

        Args:
            file (str): Path to the VCF file.
            packed (bool): If True, returns a 2-bit packed torch.Tensor. Defaults to False.
            chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        log.info("    Input format is VCF.")

        if not packed:
            G, N, M = read_vcf_file(
                file,
                chunk_size=chunk_size,
                chromosome_mode=chromosome_mode,
                autosome_count=autosome_count,
            )
            return np.ascontiguousarray(G), N, M
        else:
            log.info("        Reading VCF in packed 2-bit format for GPU use.")
            G_packed_np, N, M = read_vcf_file_packed(
                file,
                chunk_size=chunk_size,
                chromosome_mode=chromosome_mode,
                autosome_count=autosome_count,
            )
            G_packed = torch.from_numpy(G_packed_np)
            return G_packed, N, M

    def _read_pgen(self, file: str, packed: bool, chunk_size: int, chromosome_mode: str, autosome_count: int) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Internal reader for PLINK PGEN files.

        Args:
            file (str): Path to the PGEN file (without extension or with .pgen).
            packed (bool): If True, returns a 2-bit packed torch.Tensor. Defaults to False.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        log.info("    Input format is PGEN.")
        import pgenlib as pg

        with pg.PgenReader(str(file).encode()) as pgen_reader:
            num_vars = pgen_reader.get_variant_ct()
            num_samples = pgen_reader.get_raw_sample_ct()

        base_path = self._get_base_path(file)
        pvar_file = Path(base_path + ".pvar")
        bim_file = Path(base_path + ".bim")
        if pvar_file.exists():
            var_file = pvar_file
        elif bim_file.exists():
            var_file = bim_file
        else:
            log.error(f"    Error: Variant file (.pvar or .bim) not found for {base_path}")
            sys.exit(1)

        keep_mask = []
        with open(var_file) as vf:
            for line in vf:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                keep_mask.append(self._keep_chromosome(parts[0], chromosome_mode, autosome_count))
        keep_mask = np.array(keep_mask, dtype=bool)

        assert len(keep_mask) == num_vars, f"Variant file line count {len(keep_mask)} does not match PGEN variant count {num_vars}!"

        skipped = len(keep_mask) - keep_mask.sum()
        self._log_chromosome_filter(skipped, chromosome_mode, autosome_count)

        keep_idxs = np.flatnonzero(keep_mask).astype(np.uint32)
        M = keep_idxs.size
        N = num_samples

        if packed:
            log.info("        Reading PGEN in packed 2-bit format for GPU use.")
            M_bytes = (M + 3) // 4
            G_packed = torch.zeros((M_bytes, N), dtype=torch.uint8)
            variants_per_chunk = max(4, (chunk_size // 4) * 4)
            G_chunk = np.empty((variants_per_chunk, N), dtype=np.int8)

            with pg.PgenReader(str(file).encode()) as pgen_reader:
                for start in range(0, M, variants_per_chunk):
                    stop = min(start + variants_per_chunk, M)
                    chunk_len = stop - start
                    packed_start = start // 4
                    packed_len = (chunk_len + 3) // 4
                    G_chunk_view = G_chunk[:chunk_len]

                    pgen_reader.read_list(keep_idxs[start:stop], G_chunk_view)
                    replace_missing_with_three(G_chunk_view)
                    pack_genotypes(
                        G_chunk_view.view(np.uint8).ctypes.data,
                        G_packed[packed_start].data_ptr(),
                        chunk_len,
                        N,
                        packed_len,
                    )
            return G_packed, N, M

        G_raw = np.empty((M, N), dtype=np.int8)
        with pg.PgenReader(str(file).encode()) as pgen_reader:
            if M > 0:
                pgen_reader.read_list(keep_idxs, G_raw)
        replace_missing_with_three(G_raw)
        G = G_raw.view(np.uint8)

        return G, N, M

    def _check_files_exist(self, file: str, extensions: list[str], match_any: bool = False):
        """
        Description:
        Check if required files exist.

        Args:
            file (str): Path to the genotype file.
            extensions (list[str]): List of extensions to check for.
            match_any (bool): If True, check if any of the extensions exist. Defaults to False.

        Returns:
            None
        """
        base_path = self._get_base_path(file)

        if match_any:
            if not any(Path(base_path + ext).exists() for ext in extensions):
                log.error(f"    Error: Could not find any of these files: {extensions} for {base_path}")
                sys.exit(1)
        else:
            missing = [base_path + ext for ext in extensions if not Path(base_path + ext).exists()]
            if missing:
                log.error(f"    Error: Required files missing: {', '.join(missing)}")
                sys.exit(1)

    def read_data(self, file: str, packed: bool, chunk_size: int, chromosome_mode: str, autosome_count: int) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Public wrapper to read genotype data from various formats (BED, VCF).
        Automatically detects format based on file extension.

        Args:
            file (str): Path to the genotype file.
            packed (bool): If True, returns a 2-bit packed torch.Tensor (BED, PGEN, VCF). Defaults to False.
            chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.
            chromosome_mode (str): "all" to keep all chromosomes or "autosomes" to keep 1..autosome_count.
            autosome_count (int): Number of autosomes when chromosome_mode is "autosomes".

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        file_path = Path(file)
        file_extensions = file_path.suffixes
        start = time.time()

        if chromosome_mode not in {"all", "autosomes"}:
            raise ValueError("chromosome_mode must be 'all' or 'autosomes'")
        if autosome_count < 1:
            raise ValueError("autosome_count must be at least 1")

        if '.bed' in file_extensions:
            self._check_files_exist(file, ['.bed', '.fam', '.bim'])
            G, N, M = self._read_bed(file, packed, chunk_size, chromosome_mode, autosome_count)
        elif '.vcf' in file_extensions:
            self._check_files_exist(file, ['.vcf', '.vcf.gz'], match_any=True)
            G, N, M = self._read_vcf(file, packed, chunk_size, chromosome_mode, autosome_count)
        elif '.pgen' in file_extensions:
            self._check_files_exist(file, ['.pgen', '.psam', '.pvar'])
            G, N, M = self._read_pgen(file, packed, chunk_size, chromosome_mode, autosome_count)
        else:
            log.error("    Invalid format. Unrecognized file format. Make sure file ends with .bed, .pgen or .vcf .")
            sys.exit(1)

        if not packed:
            mean_val = get_mean_unpacked(G)
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (unpacked).")
                flip_unpacked(G)
        else:
            M_bytes = G.shape[0]
            mean_val = get_mean_packed(G.data_ptr(), M, N, M_bytes)
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (packed).")
                flip_packed(G.data_ptr(), M, N, M_bytes)

        end = time.time()
        log.info(f"        Total time for reading={end - start:.3f}s")

        return G, N, M
