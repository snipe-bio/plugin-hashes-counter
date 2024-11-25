import click
import logging
import os
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from ._hashes_counter_impl import HashesCounter, WeightedHashesCounter, WeightedHashesCounterUncapped, SamplesKmerDosageHybridCounter
from snipe import SnipeSig, SigType

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)  

formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

logger.propagate = False

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  
for handler in root_logger.handlers:
    handler.setFormatter(formatter)
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
    '--min-abund',
    '-m',
    type=int,
    default=None,
    help='Minimum abundance threshold to retain k-mers.',
)
@click.option(
    '--weighted',
    is_flag=True,
    default=False,
    help='Use weighted hashes counter.',
)
@click.option(
    '--uncapped',
    is_flag=True,
    default=False,
    help='Use uncapped weighted hashes counter.',
)
@click.option(
    '--hybrid',
    is_flag=True,
    default=False,
    help='Use hybrid hashes counter (sample freq + kmer dosage).',
)
def hashes_counter(
    signature_paths: List[str],
    samples_from_file: str,
    output: str,
    name: str,
    min_abund: int,
    weighted: bool,
    uncapped: bool,
    hybrid: bool,
):
    """
    Snipe plugin for high-throughput counting of k-mers.
    SIGNATURE_PATHS can include wildcards (e.g., *.sig, *.zip).
    """
    try:
        
        all_signature_paths = list(signature_paths)
        
        if samples_from_file:
            logger.debug(f"Reading signature paths from file: {samples_from_file}")
            with open(samples_from_file, 'r') as f:
                file_paths = [line.strip() for line in f if line.strip()]
                all_signature_paths.extend(file_paths)
            logger.info(f"Added {len(file_paths)} signatures from file.")
        if not all_signature_paths:
            logger.error("No signature paths provided.")
            sys.exit(1)
        
        valid_extensions = {'.sig', '.zip'}
        output_ext = os.path.splitext(output)[1].lower()
        if output_ext not in valid_extensions:
            logger.error(f"Invalid output file extension '{output_ext}'. Must be one of {valid_extensions}.")
            sys.exit(1)
        logger.info(f"Counting hashes from {len(all_signature_paths)} signatures.")
        
        if weighted:
            if uncapped:
                logger.info("Using uncapped WeightedHashesCounter.")
                counter = WeightedHashesCounterUncapped()
            else:
                logger.info("Using WeightedHashesCounter.")
                counter = WeightedHashesCounter()
        elif hybrid:
            logger.info("Using SamplesKmerDosageHybridCounter.")
            counter = SamplesKmerDosageHybridCounter()
        else:
            logger.info("Using HashesCounter.")
            counter = HashesCounter()
        
        first_sig_path = all_signature_paths[0]
        logger.debug(f"Processing first signature: {first_sig_path}")
        first_sig = SnipeSig(sourmash_sig=first_sig_path, sig_type=SigType.SAMPLE)
        auto_detected_scale = first_sig.scale
        auto_detected_ksize = first_sig.ksize
        if weighted or hybrid:
            counter.add_hashes(first_sig.hashes, first_sig.abundances, first_sig.mean_abundance)
        else:
            counter.add_hashes(first_sig.hashes)
        
        logger.debug(f"Detected scale: {auto_detected_scale}, Detected ksize: {auto_detected_ksize}")
        
        for sig_path in tqdm(all_signature_paths[1:], desc="Processing signatures"):
            snipe_sig = SnipeSig(sourmash_sig=sig_path, sig_type=SigType.SAMPLE)
            if snipe_sig.scale != auto_detected_scale or snipe_sig.ksize != auto_detected_ksize:
                logger.error(f"Signature '{sig_path}' has inconsistent scale or ksize.")
                sys.exit(1)
            if weighted:
                counter.add_hashes(snipe_sig.hashes, snipe_sig.abundances, snipe_sig.mean_abundance)
            else:
                counter.add_hashes(snipe_sig.hashes)
        if weighted or hybrid:
            logger.info("Rounding scores in WeightedHashesCounter.")
            skipped_hashes = counter.round_scores()
            logger.info(f"Skipped {skipped_hashes} hashes with <2 after rounding.")
        else:
            logger.info("Removing singleton k-mers.")
            count_removed = counter.remove_singletons()
            logger.info(f"Removed {count_removed} singleton k-mers, current size: {counter.size()}.")
        
        if min_abund is not None:
            logger.info(f"Applying minimum abundance filter: {min_abund}.")
            counter.keep_min_abundance(min_abund)
            logger.info(f"Kept only k-mers with abundance >= {min_abund}, current size: {counter.size()}.")
        
        if not hybrid:
            hash_to_abundance = counter.get_kmers()
            out_hashes = np.array(list(hash_to_abundance.keys()))
            out_abundances = np.array(list(hash_to_abundance.values()))
            
            logger.info("Creating output signature.")
            out_sig = SnipeSig.create_from_hashes_abundances(
                hashes=out_hashes,
                abundances=out_abundances,
                ksize=auto_detected_ksize,
                scale=auto_detected_scale,
                name=name,
            )
            
            logger.info(f"Exporting signature to: {output}")
            out_sig.export(output)
            logger.info("Signature export completed successfully.")
        else:
            hashes = np.array(counter.get_hashes())
            sample_counts = np.array(counter.get_sample_counts())
            kmer_dosages = np.array(counter.get_kmer_dosages())
            
            assert len(hashes) == len(sample_counts) == len(kmer_dosages)
            
            logger.info("Creating output signatures.")
            out_sig1 = SnipeSig.create_from_hashes_abundances(
                hashes=hashes,
                abundances=sample_counts,
                ksize=auto_detected_ksize,
                scale=auto_detected_scale,
                name=f"{name}_sample_counts",
            )
            out_sig2 = SnipeSig.create_from_hashes_abundances(
                hashes=hashes,
                abundances=kmer_dosages,
                ksize=auto_detected_ksize,
                scale=auto_detected_scale,
                name=f"{name}_kmer_dosages",
            )
            
            logger.info(f"Exporting signatures to: {output}")
            out_sig1.export(output.replace('.sig', '_sample_counts.sig'))
            out_sig2.export(output.replace('.sig', '_kmer_dosages.sig'))
            logger.info("Signatures export completed successfully")
            
            
            
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)
            
