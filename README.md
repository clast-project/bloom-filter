# Clast.BloomFilter

Parquet-format Split Block Bloom Filter (SBBF) with a generic-key facade. AVX2 and ARM NEON accelerated, with a scalar fallback.

[![NuGet](https://img.shields.io/nuget/v/Clast.BloomFilter.svg)](https://www.nuget.org/packages/Clast.BloomFilter/)
[![CI](https://github.com/clast-project/bloom-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/clast-project/bloom-filter/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/clast-project/bloom-filter/blob/main/LICENSE)

## Overview

`Clast.BloomFilter` provides a Split Block Bloom Filter laid out exactly the way the [Apache Parquet specification](https://github.com/apache/parquet-format/blob/master/BloomFilter.md) does, so the bytes round-trip with Parquet column-chunk bloom filters and other SBBF implementations. Each block is 256 bits (8 × `uint32`); a probe touches a single block so the entire query fits in one cache line.

- **Two layers.** `SplitBlockBloomFilter` / `SplitBlockBloomFilterBuilder` work in raw bytes and pre-hashed `ulong` keys — the surface you serialize. `BloomFilter<T>` / `BloomFilterBuilder<T>` wrap it with an `IHash64<T>` for typed, in-process use over arbitrary key types.
- **SIMD accelerated.** AVX2 on x86-64 and ARM NEON on ARM64 take the fast path; a portable scalar fallback runs everywhere else. Hashing is xxHash64.
- **Pluggable hashing.** `Hash64.Default<T>()` mirrors `EqualityComparer<T>.Default`, with specializations for the common primitives and a `GetHashCode`-based fallback for anything else.

## Example

```csharp
using Clast.BloomFilter;

// Build a typed filter sized for ~1000 distinct keys at 1% FPP.
var builder = BloomFilterBuilder<string>.WithCapacity(
    expectedDistinct: 1000, fpp: 0.01, maxBytes: 1 << 20);

foreach (var key in keys)
    builder.Add(key);

var filter = builder.Build();

if (filter.MightContain("foo"))
{
    // Definitely-not vs. probably-yes — go look in the real index.
}

// Serialize the raw bitset to disk in Parquet SBBF wire format.
File.WriteAllBytes("filter.bin", filter.Inner.Data.ToArray());

// Round-trip.
var bytes = File.ReadAllBytes("filter.bin");
var loaded = new BloomFilter<string>(new SplitBlockBloomFilter(bytes));
```

## Target frameworks

- `netstandard2.0`
- `net8.0`
- `net10.0`

The SIMD-accelerated paths require `net8.0` or later (they use the `System.Runtime.Intrinsics` APIs). The `netstandard2.0` build uses the scalar implementation on every platform.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
