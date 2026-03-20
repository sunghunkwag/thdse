#!/usr/bin/env python3
"""CLI entry point for the structural discovery pipeline.

Usage:
    python src/discover.py --corpus ./my_projects/ --output ./report/
    python src/discover.py --stdlib --output ./report/
    python src/discover.py --corpus ./proj1 ./proj2 --output ./report/
    python src/discover.py --mode diff FILE_A FILE_B
    python src/discover.py --mode refactor --corpus ./projects/
    python src/discover.py --mode synthesis --corpus ./projects/
    python src/discover.py --mode self-diagnostic --output ./report/
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _init_pipeline(args):
    """Initialize the shared pipeline components.

    Returns:
        Tuple of (arena, projector, synthesizer, hdc_core).
    """
    try:
        import hdc_core
    except ImportError:
        print("ERROR: hdc_core not compiled. Run `maturin develop` in src/hdc_core/")
        sys.exit(1)

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer

    arena = hdc_core.FhrrArena(args.capacity, args.dimension)
    projector = IsomorphicProjector(arena, args.dimension)
    synthesizer = AxiomaticSynthesizer(
        arena, projector, resonance_threshold=args.threshold,
    )
    return arena, projector, synthesizer, hdc_core


def _ingest_corpus(args, synthesizer, verbose):
    """Ingest corpus from CLI args.

    Returns:
        corpus_name string.
    """
    from src.corpus.ingester import BatchIngester
    from src.corpus.stdlib_corpus import StdlibCorpus

    ingester = BatchIngester(synthesizer, verbose=verbose)

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

    if args.save:
        if verbose:
            print(f"Saving corpus to {args.save}")
        ingester.save(args.save)

    return corpus_name


def _run_analysis(args, arena, projector, synthesizer, corpus_name, verbose):
    """Run the standard analysis mode (original behavior)."""
    from src.decoder.constraint_decoder import ConstraintDecoder
    from src.analysis.resonance_analyzer import ResonanceAnalyzer
    from src.analysis.report import ReportGenerator

    total_axioms = synthesizer.store.count()
    if verbose:
        print(f"\nTotal axioms: {total_axioms}")

    if total_axioms < 2:
        print("ERROR: Need at least 2 axioms for analysis. Check your corpus.")
        sys.exit(1)

    if verbose:
        print("Computing pairwise resonance matrix...")
    t_res = time.time()

    decoder = ConstraintDecoder(
        arena, projector, args.dimension,
        activation_threshold=0.04,
    )
    analyzer = ResonanceAnalyzer(synthesizer, decoder=decoder)
    analyzer.compute_resonance()

    t_res_done = time.time()
    stats_dict = analyzer.resonance_stats()

    if verbose:
        print(f"  Resonance computed in {t_res_done - t_res:.1f}s")
        print(f"  Mean: {stats_dict['mean']:.6f}, Std: {stats_dict['std']:.6f}")
        print(f"  Min: {stats_dict['min']:.6f}, Max: {stats_dict['max']:.6f}")

    if verbose:
        print("Running structural analysis...")

    top_pairs = analyzer.top_resonant_pairs(k=5)
    if verbose and top_pairs:
        print("\n  Top 5 most resonant pairs:")
        for a, b, r in top_pairs:
            print(f"    {r:.6f}  {a}  <->  {b}")

    outliers = analyzer.bottom_resonant_axioms(k=5)
    if verbose and outliers:
        print("\n  Top 5 structural outliers (lowest mean resonance):")
        for sid, mr in outliers:
            print(f"    {mr:.6f}  {sid}")

    fingerprints = analyzer.interpret_cliques(min_size=3)
    if verbose:
        print(f"\n  Cliques of size >= 3: {len(fingerprints)}")
        for fp in fingerprints[:5]:
            print(f"    [{len(fp.clique)} members] {fp.description}")

    bridges = analyzer.cross_project_bridges(threshold=0.8)
    if verbose and bridges:
        print(f"\n  Cross-project bridges (resonance >= 0.8): {len(bridges)}")
        for a, b, r in bridges[:5]:
            print(f"    {r:.6f}  {a}  <->  {b}")

    if verbose:
        print(f"\nGenerating report in {args.output}/")

    report_gen = ReportGenerator(
        analyzer, dimension=args.dimension, corpus_name=corpus_name,
    )
    json_path, md_path = report_gen.generate(args.output)

    if verbose:
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")


def _run_diff(args, arena, projector, verbose):
    """Run structural diff between two files."""
    from src.analysis.structural_diff import StructuralDiffEngine

    if not args.diff_files or len(args.diff_files) != 2:
        print("ERROR: --mode diff requires exactly 2 file paths")
        sys.exit(1)

    file_a, file_b = args.diff_files
    for f in (file_a, file_b):
        if not os.path.isfile(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    engine = StructuralDiffEngine(arena, projector)
    layer_sim = engine.compare_file_paths(file_a, file_b)

    print(f"\nStructural Diff: {file_a} <-> {file_b}")
    print(f"  AST similarity:      {layer_sim.sim_ast:.6f}")
    if layer_sim.sim_cfg is not None:
        print(f"  CFG similarity:      {layer_sim.sim_cfg:.6f}")
    if layer_sim.sim_data is not None:
        print(f"  Data-dep similarity: {layer_sim.sim_data:.6f}")
    print(f"  Final similarity:    {layer_sim.sim_final:.6f}")
    print(f"  Dominant layer:      {layer_sim.dominant_layer}")
    print(f"  Diagnosis:           {layer_sim.diagnosis}")

    # Write JSON output if requested
    result = {
        "file_a": file_a,
        "file_b": file_b,
        "sim_ast": round(layer_sim.sim_ast, 6),
        "sim_cfg": round(layer_sim.sim_cfg, 6) if layer_sim.sim_cfg is not None else None,
        "sim_data": round(layer_sim.sim_data, 6) if layer_sim.sim_data is not None else None,
        "sim_final": round(layer_sim.sim_final, 6),
        "dominant_layer": layer_sim.dominant_layer,
        "diagnosis": layer_sim.diagnosis,
    }
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "diff_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"\n  JSON: {json_path}")


def _run_refactor(args, arena, projector, synthesizer, verbose):
    """Scan corpus for refactoring candidates."""
    from src.analysis.refactoring_detector import RefactoringDetector

    total_axioms = synthesizer.store.count()
    if total_axioms < 2:
        print("ERROR: Need at least 2 axioms for refactoring scan.")
        sys.exit(1)

    if verbose:
        print("Scanning corpus for refactoring candidates...")

    detector = RefactoringDetector(arena, projector, synthesizer)
    candidates = detector.scan()

    if verbose:
        print(f"\n  Found {len(candidates)} refactoring candidate(s):")
        for c in candidates[:10]:
            print(f"    [{c.kind}] {c.file_a} <-> {c.file_b}")
            print(f"      {c.recommendation}")

    report = detector.to_json(candidates)
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "refactoring_candidates.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(f"\n  JSON: {json_path}")


def _run_synthesis(args, arena, projector, synthesizer, verbose):
    """Full synthesis pipeline with the new sub-tree decoder."""
    from src.decoder.constraint_decoder import ConstraintDecoder
    from src.decoder.subtree_vocab import SubTreeVocabulary

    total_axioms = synthesizer.store.count()
    if total_axioms < 2:
        print("ERROR: Need at least 2 axioms for synthesis.")
        sys.exit(1)

    if verbose:
        print("Building sub-tree vocabulary from corpus...")

    # Build sub-tree vocabulary from ingested source files
    vocab = SubTreeVocabulary()
    # Ingest source from corpus directories if available
    if args.corpus:
        for corpus_dir in args.corpus:
            for root, _dirs, files in os.walk(corpus_dir):
                for fname in files:
                    if fname.endswith(".py"):
                        vocab.ingest_file(os.path.join(root, fname))

    if verbose:
        summary = vocab.summary()
        print(f"  Sub-tree vocabulary: {summary['unique_subtrees']} unique patterns")

    # Project sub-tree atoms
    projected = vocab.project_all(arena, projector)
    if verbose:
        print(f"  Projected {projected} sub-tree atoms")

    # Create decoder with sub-tree vocabulary
    decoder = ConstraintDecoder(
        arena, projector, args.dimension,
        activation_threshold=0.04,
        subtree_vocab=vocab,
    )

    if verbose:
        print("Running synthesis pipeline...")

    synthesizer.compute_resonance()
    cliques = synthesizer.extract_cliques(min_size=2)

    results = []
    for clique in cliques[:5]:  # Limit to top 5 cliques
        synth_proj = synthesizer.synthesize_from_clique(clique)
        source = decoder.decode_to_source(synth_proj)
        if source:
            results.append({
                "clique": clique,
                "decoded_source": source,
                "entropy": synthesizer.compute_synthesis_entropy(synth_proj),
            })
            if verbose:
                print(f"\n  Clique {clique}:")
                print(f"    Decoded: {source[:200]}...")

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "synthesis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"syntheses": results}, f, indent=2)
    if verbose:
        print(f"\n  Synthesized {len(results)} program(s)")
        print(f"  JSON: {json_path}")


def _run_self_diagnostic(args, arena, projector, synthesizer, verbose):
    """Ouroboros self-analysis mode."""
    from src.decoder.constraint_decoder import ConstraintDecoder
    from src.decoder.subtree_vocab import SubTreeVocabulary

    if verbose:
        print("Running Ouroboros self-diagnostic...")

    # Build sub-tree vocab from engine's own source
    vocab = SubTreeVocabulary()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    for root, _dirs, files in os.walk(src_dir):
        for fname in files:
            if fname.endswith(".py"):
                vocab.ingest_file(os.path.join(root, fname))

    projected = vocab.project_all(arena, projector)
    if verbose:
        summary = vocab.summary()
        print(f"  Self sub-tree vocabulary: {summary['unique_subtrees']} patterns, {projected} projected")

    decoder = ConstraintDecoder(
        arena, projector, args.dimension,
        activation_threshold=0.04,
        subtree_vocab=vocab,
    )

    # Step 1: Self-diagnostic (structural analysis of own modules)
    diagnostic = synthesizer.run_self_diagnostic(
        project_root=project_root,
        output_path=args.output,
    )

    if verbose:
        print(f"\n  Self-axioms: {diagnostic.get('total_self_axioms', 0)}")
        dupes = diagnostic.get("structural_duplicates", [])
        print(f"  Internal structural duplicates: {len(dupes)}")
        for d in dupes[:3]:
            print(f"    {d['file_a']} <-> {d['file_b']} (AST={d['sim_ast']:.2f})")
        outliers = diagnostic.get("structural_outliers", [])
        if outliers:
            print(f"  Most unique module: {outliers[0]['file']}")
        for rec in diagnostic.get("recommendations", []):
            print(f"  Recommendation: {rec}")

    # Step 2: Synthesis quality self-test
    if verbose:
        print("\nRunning synthesis quality self-test...")

    quality = synthesizer.run_synthesis_quality_test(decoder=decoder)

    if verbose:
        if quality.get("round_trip_fidelity") is not None:
            print(f"  Round-trip fidelity: {quality['round_trip_fidelity']:.6f}")
            print(f"  Above threshold: {quality['fidelity_above_threshold']}")
        print(f"  Diagnosis: {quality.get('diagnosis', quality.get('error', 'N/A'))}")

    # Save quality results
    os.makedirs(args.output, exist_ok=True)
    quality_path = os.path.join(args.output, "synthesis_quality.json")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    if verbose:
        print(f"\n  self_diagnostic.json: {os.path.join(args.output, 'self_diagnostic.json')}")
        print(f"  synthesis_quality.json: {quality_path}")


def main() -> None:
    """Run the structural discovery pipeline."""
    parser = argparse.ArgumentParser(
        description="THDSE Structural Discovery Pipeline: "
        "Ingest Python code, map to VSA space, and surface structural patterns.",
    )
    parser.add_argument(
        "--mode",
        choices=["analysis", "diff", "refactor", "synthesis", "self-diagnostic"],
        default="analysis",
        help="Pipeline mode: analysis (default), diff, refactor, synthesis, self-diagnostic.",
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
    parser.add_argument(
        "diff_files",
        nargs="*",
        help="File paths for --mode diff (exactly 2 required).",
    )

    args = parser.parse_args()

    mode = args.mode
    verbose = not args.quiet

    # Validate mode-specific args
    if mode == "diff":
        if not args.diff_files or len(args.diff_files) != 2:
            parser.error("--mode diff requires exactly 2 file paths as positional arguments")
    elif mode == "self-diagnostic":
        pass  # No corpus required
    elif mode in ("analysis", "refactor", "synthesis"):
        if not args.corpus and not args.stdlib and not args.load:
            parser.error("Specify --corpus, --stdlib, or --load")

    t_start = time.time()

    # Initialize pipeline
    arena, projector, synthesizer, hdc_core = _init_pipeline(args)

    if verbose:
        print(f"Initializing pipeline (d={args.dimension}, mode={mode})")

    # Ingest corpus (not needed for diff or self-diagnostic)
    corpus_name = "corpus"
    if mode not in ("diff", "self-diagnostic"):
        corpus_name = _ingest_corpus(args, synthesizer, verbose)

        total_axioms = synthesizer.store.count()
        if verbose:
            print(f"\nTotal axioms: {total_axioms}")

    # Dispatch to mode handler
    if mode == "analysis":
        _run_analysis(args, arena, projector, synthesizer, corpus_name, verbose)

    elif mode == "diff":
        _run_diff(args, arena, projector, verbose)

    elif mode == "refactor":
        _run_refactor(args, arena, projector, synthesizer, verbose)

    elif mode == "synthesis":
        _run_synthesis(args, arena, projector, synthesizer, verbose)

    elif mode == "self-diagnostic":
        _run_self_diagnostic(args, arena, projector, synthesizer, verbose)

    t_end = time.time()
    if verbose:
        print(f"\nDone in {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
