# ZstdInflate.jl

A pure Julia implementation of [Zstandard](https://facebook.github.io/zstd/) decompression, with both in-memory and streaming interfaces, as specified in [RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878).

> **Note:** This package was developed in collaboration with an AI assistant (Claude). The implementation has not yet been rigorously vetted and should be treated accordingly — thorough review and testing is recommended before use in production.

## Installation

```julia
using Pkg
Pkg.add("ZstdInflate")
```

## Usage

### In-memory decompression

```julia
using ZstdInflate

data = inflate_zstd(compressed::Vector{UInt8})  # decompress bytes
data = inflate_zstd(filename::AbstractString)   # decompress a file
```

### Streaming decompression

```julia
open(filename) do io
    stream = InflateZstdStream(io)
    data = read(stream)
end
```

`ZstandardStream` wraps any readable `IO` object and decompresses on the fly.

### Dictionaries

Parse the dictionary once, then pass it to any decompression call:

```julia
dict_bytes = read("my.dict")
dict = parse(ZstdDict, dict_bytes)                        # auto-detects structured vs raw content
# or: parse(ZstdDict, dict_bytes; raw_content=true)       # force raw content mode

data   = inflate_zstd(compressed; dict=dict)
text   = inflate_zstd("file.zst"; dict=dict)

open("file.zst") do io
    stream = InflateZstdStream(io; dict=dict)
    data = read(stream)
end
```

## Acknowledgements

This package was inspired by [Inflate.jl](https://github.com/GunnarFarneback/Inflate.jl), a pure Julia Deflate/zlib/gzip decompressor, which served as a model for the package structure and approach.

## Why choose ZstdInflate.jl over CodecZstd.jl

- Pure Julia — no binary dependencies, works anywhere Julia does.
- Competitive performance for text and other compressible data.

## Why choose CodecZstd.jl over ZstdInflate.jl

- Need compression (ZstdInflate.jl is decompression only).
- Want a TranscodingStreams-compatible interface.
- Want a battle-tested C library backing.

# Performance Comparison

TODO
