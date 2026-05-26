// Copyright (c) clast-project. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Clast.BloomFilter;

/// <summary>
/// Typed probe-only bloom filter. Wraps a
/// <see cref="SplitBlockBloomFilter"/> and an <see cref="IHash64{T}"/>
/// to give a clean <c>MightContain(T)</c> surface without forcing
/// callers to think about hashing or byte-level encoding.
/// </summary>
public sealed class BloomFilter<T>
{
    private readonly SplitBlockBloomFilter _inner;
    private readonly IHash64<T> _hash;

    /// <summary>Wraps an existing byte-level filter with the given hash function.</summary>
    public BloomFilter(SplitBlockBloomFilter inner, IHash64<T>? hash = null)
    {
        if (inner is null) throw new ArgumentNullException(nameof(inner));
        _inner = inner;
        _hash = hash ?? Hash64.Default<T>();
    }

    /// <summary>The underlying byte-level filter, for serialization.</summary>
    public SplitBlockBloomFilter Inner => _inner;

    /// <summary>Tests whether <paramref name="value"/> might be present.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool MightContain(T value) => _inner.MightContainHash(_hash.Hash(value));

    /// <summary>
    /// Probes many values at once, writing one result per value to
    /// <paramref name="results"/>. Hashing is done up front so the probes run as
    /// a tight, unrolled batch (see
    /// <see cref="SplitBlockBloomFilter.MightContainHash(ReadOnlySpan{ulong}, Span{bool})"/>).
    /// </summary>
    public void MightContain(ReadOnlySpan<T> values, Span<bool> results)
    {
        if (results.Length < values.Length)
            throw new ArgumentException(
                "Results span must be at least as long as the values span.", nameof(results));

        int n = values.Length;
        if (n == 0) return;

        ulong[] hashes = ArrayPool<ulong>.Shared.Rent(n);
        try
        {
            for (int i = 0; i < n; i++)
                hashes[i] = _hash.Hash(values[i]);
            _inner.MightContainHash(hashes.AsSpan(0, n), results);
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(hashes);
        }
    }
}
