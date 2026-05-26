// Copyright (c) clast-project. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Clast.BloomFilter;

BenchmarkRunner.Run<ProbeBenchmarks>();

/// <summary>
/// Compares one-at-a-time probing against the unrolled batch probe across
/// hit/miss ratios. The miss-heavy case is the row-group-skipping workload
/// where the block-load latency dominates and batching's overlapping loads win.
/// </summary>
[MemoryDiagnoser]
public class ProbeBenchmarks
{
    private SplitBlockBloomFilter _filter = null!;
    private ulong[] _hashes = null!;
    private bool[] _results = null!;

    /// <summary>Fraction of probes that hit an inserted value (0 = all misses).</summary>
    [Params(0.0, 0.5, 1.0)]
    public double HitRatio;

    [Params(10_000)]
    public int ProbeCount;

    /// <summary>
    /// Distinct inserted values, which sets filter size: 100K ≈ 120 KB
    /// (cache-resident) vs 40M ≈ 48 MB (exceeds LLC, so probes hit DRAM and the
    /// batch's overlapping loads can hide latency).
    /// </summary>
    [Params(100_000, 40_000_000)]
    public int Distinct;

    [GlobalSetup]
    public void Setup()
    {
        int distinct = Distinct;
        var builder = new SplitBlockBloomFilterBuilder(
            SplitBlockBloomFilterBuilder.OptimalNumBytes(distinct, 0.01, 1 << 28));

        var rng = new Random(1);
        var inserted = new ulong[distinct];
        for (int i = 0; i < distinct; i++)
        {
            inserted[i] = NextHash(rng);
            builder.AddHash(inserted[i]);
        }
        _filter = builder.Build();

        _hashes = new ulong[ProbeCount];
        int hits = (int)(ProbeCount * HitRatio);
        for (int i = 0; i < ProbeCount; i++)
            _hashes[i] = i < hits ? inserted[rng.Next(distinct)] : NextHash(rng);
        Shuffle(_hashes, rng); // interleave hits and misses

        _results = new bool[ProbeCount];
    }

    [Benchmark(Baseline = true)]
    public int Single()
    {
        int count = 0;
        var hashes = _hashes;
        for (int i = 0; i < hashes.Length; i++)
            if (_filter.MightContainHash(hashes[i]))
                count++;
        return count;
    }

    [Benchmark]
    public int Batch()
    {
        _filter.MightContainHash(_hashes, _results);
        int count = 0;
        var results = _results;
        for (int i = 0; i < results.Length; i++)
            if (results[i])
                count++;
        return count;
    }

    private static ulong NextHash(Random rng)
    {
        var b = new byte[8];
        rng.NextBytes(b);
        return BitConverter.ToUInt64(b, 0);
    }

    private static void Shuffle(ulong[] a, Random rng)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
