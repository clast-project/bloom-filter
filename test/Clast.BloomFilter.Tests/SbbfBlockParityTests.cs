// Copyright (c) clast-project. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.

#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif

namespace Clast.BloomFilter.Tests;

/// <summary>
/// Cross-checks that the SIMD probe/insert kernels produce results
/// bit-identical to the scalar reference, and that the batch probe API agrees
/// with single-value probing. Mirrors the cross-target diff test added in
/// Apache Arrow PR #50030.
/// </summary>
public class SbbfBlockParityTests
{
    private const int BlockBytes = SplitBlockBloomFilter.BytesPerBlock; // 32

    private static ulong NextHash(Random rng)
    {
        var b = new byte[8];
        rng.NextBytes(b);
        return BitConverter.ToUInt64(b, 0);
    }

#if NET8_0_OR_GREATER
    [Fact]
    public void Probe_SimdKernels_MatchScalar()
    {
        if (!Avx2.IsSupported && !AdvSimd.IsSupported)
            return; // No SIMD kernel on this machine; scalar is the only path.

        var rng = new Random(12345);
        var block = new byte[BlockBytes];

        // Random blocks at every density — empty, sparse, dense, near-full.
        for (int iter = 0; iter < 20_000; iter++)
        {
            rng.NextBytes(block);
            uint key = (uint)NextHash(rng);

            bool expected = SbbfBlock.BlockProbeScalar(block, 0, key);

            if (Avx2.IsSupported)
                Assert.Equal(expected, SbbfBlock.BlockProbeAvx2(block, 0, key));
            if (AdvSimd.IsSupported)
                Assert.Equal(expected, SbbfBlock.BlockProbeNeon(block, 0, key));
        }
    }

    [Fact]
    public void Probe_SimdKernels_MatchScalar_ForGuaranteedHits()
    {
        if (!Avx2.IsSupported && !AdvSimd.IsSupported)
            return;

        var rng = new Random(54321);

        // A key inserted into a block must probe-hit (true) on every kernel —
        // exercises the all-lanes-set path that random blocks rarely reach.
        for (int iter = 0; iter < 20_000; iter++)
        {
            var block = new byte[BlockBytes];
            rng.NextBytes(block);
            uint key = (uint)NextHash(rng);

            SbbfBlock.BlockInsertScalar(block, 0, key);

            Assert.True(SbbfBlock.BlockProbeScalar(block, 0, key));
            if (Avx2.IsSupported)
                Assert.True(SbbfBlock.BlockProbeAvx2(block, 0, key));
            if (AdvSimd.IsSupported)
                Assert.True(SbbfBlock.BlockProbeNeon(block, 0, key));
        }
    }

    [Fact]
    public void Insert_SimdKernels_MatchScalar()
    {
        if (!Avx2.IsSupported && !AdvSimd.IsSupported)
            return;

        var rng = new Random(67890);
        var seed = new byte[BlockBytes];

        for (int iter = 0; iter < 20_000; iter++)
        {
            rng.NextBytes(seed); // start from a pre-populated block
            uint key = (uint)NextHash(rng);

            var scalarBlock = (byte[])seed.Clone();
            SbbfBlock.BlockInsertScalar(scalarBlock, 0, key);

            if (Avx2.IsSupported)
            {
                var avx2Block = (byte[])seed.Clone();
                SbbfBlock.BlockInsertAvx2(avx2Block, 0, key);
                Assert.Equal(scalarBlock, avx2Block);
            }
            if (AdvSimd.IsSupported)
            {
                var neonBlock = (byte[])seed.Clone();
                SbbfBlock.BlockInsertNeon(neonBlock, 0, key);
                Assert.Equal(scalarBlock, neonBlock);
            }
        }
    }
#endif

    [Fact]
    public void BatchProbe_MatchesSingleProbe_AndHasNoFalseNegatives()
    {
        var builder = new SplitBlockBloomFilterBuilder(
            SplitBlockBloomFilterBuilder.OptimalNumBytes(2000, 0.01, 1 << 20));

        var rng = new Random(999);
        var present = new ulong[2000];
        for (int i = 0; i < present.Length; i++)
        {
            present[i] = NextHash(rng);
            builder.AddHash(present[i]);
        }

        var filter = builder.Build();

        // Mix inserted hashes (hits) with random hashes (mostly misses), plus a
        // tail that isn't a multiple of 4 to cover the unrolled remainder.
        var hashes = new ulong[4099];
        for (int i = 0; i < hashes.Length; i++)
            hashes[i] = (i % 3 == 0) ? present[i % present.Length] : NextHash(rng);

        var batch = new bool[hashes.Length];
        filter.MightContainHash(hashes, batch);

        for (int i = 0; i < hashes.Length; i++)
            Assert.Equal(filter.MightContainHash(hashes[i]), batch[i]);

        // No false negatives: every inserted value must read back as present.
        for (int i = 0; i < present.Length; i++)
            Assert.True(filter.MightContainHash(present[i]));
    }

    [Fact]
    public void BatchProbe_ThrowsWhenResultsTooShort()
    {
        var filter = new SplitBlockBloomFilterBuilder(64).Build();
        var hashes = new ulong[8];
        var results = new bool[7];
        Assert.Throws<ArgumentException>(() => filter.MightContainHash(hashes, results));
    }

    [Fact]
    public void BatchProbe_EmptyInput_IsNoOp()
    {
        var filter = new SplitBlockBloomFilterBuilder(64).Build();
        filter.MightContainHash(ReadOnlySpan<ulong>.Empty, Span<bool>.Empty);
    }

    [Fact]
    public void FacadeBatchProbe_MatchesSingleProbe()
    {
        var builder = BloomFilterBuilder<int>.WithCapacity(1000, 0.01, 1 << 20);
        for (int i = 0; i < 1000; i++)
            builder.Add(i);

        var filter = builder.Build();

        // Even values 0..1998 — the first 1000 even values overlap inserted keys.
        var values = new int[1500];
        for (int i = 0; i < values.Length; i++)
            values[i] = i; // 0..1499: 0..999 are present, 1000..1499 likely absent

        var results = new bool[values.Length];
        filter.MightContain(values, results);

        for (int i = 0; i < values.Length; i++)
            Assert.Equal(filter.MightContain(values[i]), results[i]);

        // No false negatives for inserted values.
        for (int i = 0; i < 1000; i++)
            Assert.True(results[i]);
    }
}
