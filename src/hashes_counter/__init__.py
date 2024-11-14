import click
import logging
import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm

from ._hashes_counter_impl import HashesCounter
from snipe import SnipeSig, SigType

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Default log level set to WARNING

# Create handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Stream handler set to INFO for user-facing output

# Create formatter and add to handler
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

@click.command()
@click.argument(
    'signature_paths',
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar='SIGNATURE_PATHS',
)
@click.option(
    '--samples-from-file',
    '-f',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Path to a text file containing signature paths (one per line).',
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(writable=True),
    help='Output file path. Must end with .sig or .zip.',
)
@click.option(
    '--name',
    '-n',
    required=True,
    type=str,
    help='Name of the output signature.',
)
@click.option(
    '--remove-singletons',
    is_flag=True,
    default=False,
    help='Remove singleton k-mers after counting.',
)
@click.option(
    '--min-abund',
    '-m',
    type=int,
    default=None,
    help='Minimum abundance threshold to retain k-mers.',
)
def hashes_counter(
    signature_paths: List[str],
    samples_from_file: str,
    output: str,
    name: str,
    remove_singletons: bool,
    min_abund: int,
):
    """
    Snipe plugin for high-throughput counting of k-mers.

    SIGNATURE_PATHS can include wildcards (e.g., *.sig, *.zip).
    """
    try:
        # Initialize list of signature paths
        all_signature_paths = list(signature_paths)

        # If --samples-from-file is provided, parse additional signature paths
        if samples_from_file:
            logger.debug(f"Reading signature paths from file: {samples_from_file}")
            with open(samples_from_file, 'r') as f:
                file_paths = [line.strip() for line in f if line.strip()]
                all_signature_paths.extend(file_paths)
            logger.info(f"Added {len(file_paths)} signatures from file.")

        if not all_signature_paths:
            logger.error("No signature paths provided.")
            sys.exit(1)

        # Validate output file extension
        valid_extensions = {'.sig', '.zip'}
        output_ext = os.path.splitext(output)[1].lower()
        if output_ext not in valid_extensions:
            logger.error(f"Invalid output file extension '{output_ext}'. Must be one of {valid_extensions}.")
            sys.exit(1)

        logger.info(f"Counting hashes from {len(all_signature_paths)} signatures.")

        counter = HashesCounter()

        # Initialize with the first signature
        first_sig_path = all_signature_paths[0]
        logger.debug(f"Processing first signature: {first_sig_path}")
        first_sig = SnipeSig(sourmash_sig=first_sig_path, sig_type=SigType.SAMPLE)
        auto_detected_scale = first_sig.scale
        auto_detected_ksize = first_sig.ksize
        counter.add_hashes(first_sig.hashes)
        logger.debug(f"Detected scale: {auto_detected_scale}, Detected ksize: {auto_detected_ksize}")

        # Process remaining signatures with a progress bar
        for sig_path in tqdm(all_signature_paths[1:], desc="Processing signatures"):
            snipe_sig = SnipeSig(sourmash_sig=sig_path, sig_type=SigType.SAMPLE)
            if snipe_sig.scale != auto_detected_scale or snipe_sig.ksize != auto_detected_ksize:
                logger.error(f"Signature '{sig_path}' has inconsistent scale or ksize.")
                sys.exit(1)
            counter.add_hashes(snipe_sig.hashes)

        # Optional: Remove singletons
        if remove_singletons:
            logger.info("Removing singleton k-mers.")
            count_removed = counter.remove_singletons()
            logger.info(f"Removed {count_removed} singleton k-mers, current size: {counter.size()}.")

        # Optional: Apply minimum abundance filter
        if min_abund is not None:
            logger.info(f"Applying minimum abundance filter: {min_abund}.")
            counter.keep_min_abundance(min_abund)
            logger.info(f"Kept only k-mers with abundance >= {min_abund}, current size: {counter.size()}.")

        # Retrieve k-mers and their abundances
        hash_to_abundance = counter.get_kmers()
        out_hashes = np.array(list(hash_to_abundance.keys()))
        out_abundances = np.array(list(hash_to_abundance.values()))

        # Create output signature
        logger.info("Creating output signature.")
        out_sig = SnipeSig.create_from_hashes_abundances(
            hashes=out_hashes,
            abundances=out_abundances,
            ksize=auto_detected_ksize,
            scale=auto_detected_scale,
            name=name,
        )

        # Export the signature
        logger.info(f"Exporting signature to: {output}")
        out_sig.export(output)
        logger.info("Signature export completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)
