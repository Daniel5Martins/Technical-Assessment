# Bioinformatics Mini-Pipeline

This workflow was developed as part of a technical assessment in Bioinformatics. It processes paired-end FASTQ files, aligns the reads to the hg38 human reference genome, and outlines the subsequent steps for variant calling. The pipeline is implemented using Snakemake.

> **Note:** Input FASTQ files and raw output files are not shared in this public repository. Samples were manually anonymized before running the pipeline, and a separate metadata file mapping original IDs to anonymized IDs was delivered separately.

---

# Setup & Running

## Conda / Snakemake

To install Miniconda, visit:  
[https://www.anaconda.com/docs/getting-started/miniconda/install](https://www.anaconda.com/docs/getting-started/miniconda/install)

To install Snakemake and activate the environment in a Linux OS, run:

```bash
conda create -c conda-forge -c bioconda -n snakemake snakemake
conda activate snakemake
```

Alternatively, visit [Snakemake installation guide](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

## Requirements

The pipeline expects the following files and folders in the working directory:

- `Snakefile` file
- `config.yaml` file
- `fastq/` directory
    - _Input_ `fastq.gz` _files_
- `envs/` directory
    - _Provided within this repo_
- (Optional) `reference/` directory
    - `genome.fa`

To run the pipeline from the working directory, use:

```bash
    snakemake --use-conda --cores N
```

Replace *N* with the number of CPU cores available for running the pipeline.

# Pipeline

This workflow performs a standard read processing and alignment pipeline.

1. Trimming of paired-end sequencing reads.

2. Alignment to a reference genome (hg38).

3. Sorting, indexing, and duplicate marking of BAM files

## Quality Control

Quality control (QC) reports are compiled at three key stages:

1. Before trimming, to assess initial FASTQ read quality and guide trimming requirements.
    
    - For this pipeline, the decision to trim paired-end adapters and poly-G tails was informed by the pre-trimming QC reports. 

2. After trimming, to evaluate improvements from fastp.

3. Post-alignment and BAM processing, to inspect alignment quality and PCR duplicates.

A final consolidated report, including post-trimming and alignment QC reports, is generated with the MultiQC tool.

## Reference Genome Handling

The pipeline includes a rule to automatically download the hg38 genome from Ensembl. This step is skipped if:

- A file named `genome.fa` is found in the `reference/` directory.

Or partially skipped if:

- A gzipped genome is present and its path is set in `config.yaml`.

# Tools

The tools used in each processing step were chosen due to their wide adoption, support in genomic workflows, and the interoperability with downstream tools.

| Tool       | Version | Description                                                              |
|------------|---------|--------------------------------------------------------------------------|
| FastQC     | 0.12.1  | Generates reports on FASTQ file reads.                                   |
| Fastp      | 1.0.1   | Trims raw input reads.                                                   |
| BWA-MEM    | 0.7.19  | Aligns short reads; robust default aligner for 150bp paired-end reads.   |
| Samtools   | 1.22.1  | Sorts, indexes and generates QC reports (flagstat, idxstat, stats).      |
| Picard     | 3.4.0   | Marks PCR duplicates in BAM files.                                       |
| MultiQC    | 1.30    | Compiles quality and metrics reports generated throughout the pipeline.  |

***Notes:***
- FastQC reports confirmed the use of Illumina encoding within the FASTQ files besides the standard read header format for this platform. This enabled the definition of the read group (@RG) identifier, needed for Picard and downstream analyses.

- Picard duplicates were marked but not removed to preserve downstream flexibility. By marking duplicates instead of removing them, tools such as GATK can still effectively ignore duplicate reads without altering the original BAM files.

- Versions presented in the table refer to their use for the technical assessment on 16/7/2025. Alignments (and downstream post-alignment processes) were rerun on 17/7/2025. Versions were up to date as of 15/7/2025.

# Summary of results

The presented pipeline produced the expected results, passing quality control metrics for post-trimming read quality, alignment performance, and post-alignment processing.

## Read Quality

Fastp reports for the 3 samples presented similar filtering performances, retaining ~98.5% of the read pairs. This corresponds approximately to 1.97 million read pairs. After filtering, approximately 98.84% of the bases (across the 3 samples) had a Phred quality score of at least Q20, reflecting the overall high sequencing accuracy.

Regarding the post-trimming quality, FastQC reported consistently high per-base Phred scores across the reads (mean Phred scores > 38) and a residual number of reads with a mean quality below a 30 Phred score.

The GC content distribution displays a peak around 40%, consistent with the genome-wide average, followed by a gradual decline up to approximately 60% GC content. This pattern likely reflects the enrichment of GC-rich exonic targets. Nonetheless, the majority of reads are concentrated within the expected GC content range of 40–60%.

The sequence length distribution is consistent across files and concentrated on the higher end (150bp). Both FastQC and MultiQC reports present a warning for these results without failing to meet acceptable thresholds. Some length variability is expected after trimming within acceptable limits.

Adapters and overrepresented sequences were successfully removed. The final MultiQC reflects the presence of a residual number of poly-A sequences that could be removed using the `fastp --trim_poly_x` option if needed for downstream applications. This optional step was not applied here, as the overall quality metrics remained within acceptable thresholds.

## Aligment statistics

The alignment step yielded consistent and high-quality results across all three samples. The total number of paired-end reads per sample ranged from ~1.95 to 1.96 million. Of those, 100% were successfully mapped to the reference genome after applying a minimum mapping quality threshold (`-q 20`), indicating highly specific and accurate alignments.

A very high proportion of the reads (~99.5%) were properly paired, reflecting excellent library preparation and fragment size consistency.

Picard reported approximately 1% duplicates in each sample — a relatively low rate for target enrichment protocols, where higher duplication levels are often observed due to PCR amplification.

Overall, these metrics confirm a highly successful and clean alignment, consistent with high-quality input data and appropriate preprocessing steps.

Region-specific coverage analysis (e.g., using bedtools coverage) was not performed due to the absence of a BED file defining target intervals. Global coverage metrics were assessed using samtools stats and summarized via MultiQC.

# Downstream variant calling

The specific steps following this pipeline depend, to some extent, on the goals of the variant calling. For instance:

- Exploratory applications benefit from more lenient post-alignment filtering and variant calling.

- Diagnostic applications require stricter parameters to ensure reproducibility and accuracy.

Nonetheless, GATK and, more specifically, its `HaplotypeCaller` tool represent a flexible and well-documented solution for the steps downstream to this pipeline.

## Reference Preparation

Some preparation is still needed, as GATK requires a sequence dictionary alongside the FASTA index:

```bash
gatk CreateSequenceDictionary -R ref/genome.fa -O ref/genome.dict
```

The pipeline already produces `genome.fa.fai` via `samtools faidx`, which is also required by GATK tools.

## Suggested GATK-Based Steps

Assuming sorted and duplicate-marked BAM files (as produced by the pipeline), the following steps could be taken:

1. Post-alignment filtering

    - Additional filtering can be applied using `samtools` view to retain only higher-confidence alignments.

    - For instance, setting a higher minimum mapping quality threshold (e.g., `-q` 30) or excluding residual unmapped reads (`-F` 4) may improve variant calling accuracy.

    - That said, due to the quality of the alignments already produced by the pipeline, the gains from further filtering may be modest.

2. Base Quality Score Recalibration (BQSR)

    - Optional. Typically used in larger datasets, BQSR helps model systematic errors in base quality scores.

    - It requires known variant sites and benefits from larger cohorts, so it may not be essential for this dataset.

3. Variant Calling with HaplotypeCaller

    - `HaplotypeCaller` can be run in GVCF mode per sample, followed by joint genotyping using `GenotypeGVCFs`.

    - Given that these samples likely represent a well-known and studied genome trio, joint genotyping may be particularly informative.

After variant calling, filtering is essential to distinguish true positives from artifacts. When possible, GATK's Variant Quality Score Recalibration (VQSR) is recommended, as it uses machine learning to assign a probability that each variant is real. For smaller sample sets, as in this case, where VQSR may not be feasible, hard-filtering based on annotation thresholds (e.g., QD, MQ) is an alternative.

For further guidance, including specific parameters and examples, refer to the [GATK Best Practices](https://gatk.broadinstitute.org/hc/en-us/sections/360007226651-Best-Practices-Workflows) maintained by the Broad Institute.
