#!/usr/bin/env python3
"""CLI entry point for the structural discovery pipeline.

Usage:
    python src/discover.py --corpus ./my_projects/ --output ./report/
    python src/discover.py --stdlib --output ./report/
    python src/discover.py --corpus ./proj1 ./proj2 --output ./report/
"""

import argparse
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> None:
    """Run the structural discovery pipeline."""
    parser = argparse.ArgumentParser(
        description="THDSE Structural Discovery Pipeline: "
        "Ingest Python code, map to VSA space, and surface structural patterns.",
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        help="One or more directories of Python projects to analyze.",
    )
    parser.add_argument(
        "--stdlib",
        action="store_true",
        help="Analyze the Python standard library (no downloads needed).",
    )
    parser.add_argument(
        "--output",
        default="./report",
        help="Output directory for reports (default: ./report).",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=256,
        help="Hypervector dimension (default: 256).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=2_000_000,
        help="Arena capacity in handles (default: 2000000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Resonance threshold for clique formation (default: 0.15).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum files to ingest (0 = unlimited).",
    )
    parser.add_argument(
        "--load",
        help="Load a previously saved corpus (.pkl) instead of re-ingesting.",
    )
    parser.add_argument(
        "--save",
        help="Save the ingested corpus to a .pkl file for reuse.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    if not args.corpus and not args.stdlib and not args.load:
        parser.error("Specify --corpus, --stdlib, or --load")

    verbose = not args.quiet

    # Import after arg parsing for faster --help
    try:
        import hdc_core
    except ImportError:
        print("ERROR: hdc_core not compiled. Run `maturin develop` in src/hdc_core/")
        sys.exit(1)

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
    from src.decoder.constraint_decoder import ConstraintDecoder
    from src.corpus.ingester import BatchIngester
    from src.corpus.stdlib_corpus import StdlibCorpus
    from src.analysis.resonance_analyzer import ResonanceAnalyzer
    from src.analysis.report import ReportGenerator

    t_start = time.time()

    # ── Step 1: Initialize pipeline ────────────────────────────
    if verbose:
        print(f"Initializing pipeline (d={args.dimension}, capacity={args.capacity})")

    arena = hdc_core.FhrrArena(args.capacity, args.dimension)
    projector = IsomorphicProjector(arena, args.dimension)
    synthesizer = AxiomaticSynthesizer(
        arena, projector, resonance_threshold=args.threshold,
    )

    ingester = BatchIngester(synthesizer, verbose=verbose)

    # ── Step 2: Ingest corpus ──────────────────────────────────
    if args.load:
        if verbose:
            print(f"Loading corpus from {args.load}")
        count = BatchIngester.load(args.load, synthesizer)
        if verbose:
            print(f"  Loaded {count} axioms")
        corpus_name = os.path.basename(args.load).replace(".pkl", "")
    else:
        corpus_name_parts = []

        if args.stdlib:
            if verbose:
                print("Ingesting Python standard library...")
            stdlib = StdlibCorpus(ingester)
            stats = stdlib.ingest(max_files=args.max_files)
            if verbose:
                print(f"  Stdlib: {stats.summary()}")
            corpus_name_parts.append("stdlib")

        if args.corpus:
            if verbose:
                print(f"Ingesting {len(args.corpus)} project(s)...")
            if len(args.corpus) == 1:
                stats = ingester.ingest_directory(
                    args.corpus[0],
                    project_prefix=os.path.basename(args.corpus[0]),
                )
                if verbose:
                    print(f"  {stats.summary()}")
                corpus_name_parts.append(os.path.basename(args.corpus[0]))
            else:
                stats = ingester.ingest_multiple_projects(args.corpus)
                if verbose:
                    print(f"  Total: {stats.summary()}")
                corpus_name_parts.extend(
                    os.path.basename(d) for d in args.corpus
                )

        corpus_name = "+".join(corpus_name_parts) if corpus_name_parts else "corpus"

    total_axioms = synthesizer.store.count()
    if verbose:
        print(f"\nTotal axioms: {total_axioms}")

    if total_axioms < 2:
        print("ERROR: Need at least 2 axioms for analysis. Check your corpus.")
        sys.exit(1)

    # ── Step 3: Save corpus if requested ───────────────────────
    if args.save:
        if verbose:
            print(f"Saving corpus to {args.save}")
        ingester.save(args.save)

    # ── Step 4: Compute resonance ──────────────────────────────
    if verbose:
        print("Computing pairwise resonance matrix...")
    t_res = time.time()

    decoder = ConstraintDecoder(
        arena, projector, args.dimension,
        activation_threshold=0.04,
    )
    analyzer = ResonanceAnalyzer(synthesizer, decoder=decoder)
    mat = analyzer.compute_resonance()

    t_res_done = time.time()
    stats_dict = analyzer.resonance_stats()

    if verbose:
        print(f"  Resonance computed in {t_res_done - t_res:.1f}s")
        print(f"  Mean: {stats_dict['mean']:.6f}, Std: {stats_dict['std']:.6f}")
        print(f"  Min: {stats_dict['min']:.6f}, Max: {stats_dict['max']:.6f}")

    # ── Step 5: Analysis ───────────────────────────────────────
    if verbose:
        print("Running structural analysis...")

    # Top pairs
    top_pairs = analyzer.top_resonant_pairs(k=5)
    if verbose and top_pairs:
        print("\n  Top 5 most resonant pairs:")
        for a, b, r in top_pairs:
            print(f"    {r:.6f}  {a}  <->  {b}")

    # Outliers
    outliers = analyzer.bottom_resonant_axioms(k=5)
    if verbose and outliers:
        print("\n  Top 5 structural outliers (lowest mean resonance):")
        for sid, mr in outliers:
            print(f"    {mr:.6f}  {sid}")

    # Cliques
    fingerprints = analyzer.interpret_cliques(min_size=3)
    if verbose:
        print(f"\n  Cliques of size >= 3: {len(fingerprints)}")
        for fp in fingerprints[:5]:
            print(f"    [{len(fp.clique)} members] {fp.description}")

    # Cross-project bridges
    bridges = analyzer.cross_project_bridges(threshold=0.8)
    if verbose and bridges:
        print(f"\n  Cross-project bridges (resonance >= 0.8): {len(bridges)}")
        for a, b, r in bridges[:5]:
            print(f"    {r:.6f}  {a}  <->  {b}")

    # ── Step 6: Generate report ────────────────────────────────
    if verbose:
        print(f"\nGenerating report in {args.output}/")

    report_gen = ReportGenerator(
        analyzer, dimension=args.dimension, corpus_name=corpus_name,
    )
    json_path, md_path = report_gen.generate(args.output)

    t_end = time.time()

    if verbose:
        print(f"\nDone in {t_end - t_start:.1f}s")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
