// Copyright (c) clast-project. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.

using System;
using System.IO.Hashing;
using System.Runtime.CompilerServices;

namespace Clast.BloomFilter;

/// <summary>
/// Parquet Split Block Bloom Filter (SBBF). Each block is 256 bits
/// (8 × uint32); a probe touches a single block so the entire filter
/// query fits in one cache line. Hashing is xxHash64.
/// </summary>
/// <remarks>
/// This is the byte-level surface — what gets persisted to a file
/// format. For probing arbitrary typed keys in-process, prefer the
/// <see cref="BloomFilter{T}"/> facade.
/// </remarks>
public sealed class SplitBlockBloomFilter
{
    internal const int BytesPerBlock = 32;

    private readonly byte[] _data;
    private readonly int _numBlocks;

    /// <summary>Creates a bloom filter over the given raw bitset data.</summary>
    /// <param name="data">Filter bitset. Length must be a positive multiple of 32.</param>
    public SplitBlockBloomFilter(byte[] data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length == 0 || data.Length % BytesPerBlock != 0)
            throw new ArgumentException(
                $"Bloom filter data length must be a positive multiple of {BytesPerBlock}, got {data.Length}.",
                nameof(data));

        _data = data;
        _numBlocks = data.Length / BytesPerBlock;
    }

    /// <summary>Returns the underlying bitset bytes (live reference, do not mutate).</summary>
    public ReadOnlySpan<byte> Data => _data;

    /// <summary>Tests whether the given plain-encoded value might be present.</summary>
    public bool MightContain(ReadOnlySpan<byte> value)
    {
        ulong hash = XxHash64.HashToUInt64(value);
        return MightContainHash(hash);
    }

    /// <summary>Tests whether a pre-hashed value might be present. The 64-bit hash should be uniformly distributed.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool MightContainHash(ulong hash)
    {
        uint upper = (uint)(hash >> 32);
        int blockIndex = (int)(((ulong)upper * (ulong)_numBlocks) >> 32);
        uint key = (uint)hash;

        return SbbfBlock.BlockProbe(_data, blockIndex * BytesPerBlock, key);
    }

    /// <summary>
    /// Probes many pre-hashed values at once, writing one result per hash to
    /// <paramref name="results"/>. Each <c>results[i]</c> is <see langword="true"/>
    /// if <c>hashes[i]</c> might be present and <see langword="false"/> if it is
    /// definitely absent.
    /// </summary>
    /// <remarks>
    /// Each probe is independent; the unrolled loop keeps several of their block
    /// loads in flight so the CPU's out-of-order window can overlap them. This
    /// helps most when the filter is larger than cache (probes hit DRAM) and
    /// selectivity is unpredictable — the row-group/page-skipping case — where
    /// it measures ~25% faster than a one-at-a-time loop. For small,
    /// cache-resident filters or fully hit-/miss-homogeneous inputs there is no
    /// latency to hide and the per-result write makes a single-probe loop as
    /// fast or faster; prefer <see cref="MightContainHash(ulong)"/> there.
    /// Results are bit-identical to probing each value individually.
    /// </remarks>
    public void MightContainHash(ReadOnlySpan<ulong> hashes, Span<bool> results)
    {
        if (results.Length < hashes.Length)
            throw new ArgumentException(
                "Results span must be at least as long as the hashes span.", nameof(results));

        int n = hashes.Length;
        int i = 0;

        // Unroll by 4 to widen the out-of-order window across independent probes.
        for (; i <= n - 4; i += 4)
        {
            bool r0 = MightContainHash(hashes[i]);
            bool r1 = MightContainHash(hashes[i + 1]);
            bool r2 = MightContainHash(hashes[i + 2]);
            bool r3 = MightContainHash(hashes[i + 3]);
            results[i] = r0;
            results[i + 1] = r1;
            results[i + 2] = r2;
            results[i + 3] = r3;
        }

        for (; i < n; i++)
            results[i] = MightContainHash(hashes[i]);
    }
}
