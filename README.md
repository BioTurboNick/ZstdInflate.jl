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

data = inflate_zstd(compressed::Vector{UInt8})  # decompress bytes → Vector{UInt8}
text = inflate_zstd(filename::AbstractString)   # decompress a .zst file → String
```

Multiple concatenated frames are decompressed and concatenated; skippable
frames are ignored. When the input contains two or more frames, they can be
decompressed in parallel by passing `nthreads` (defaults to
`Threads.nthreads()`):

```julia
data = inflate_zstd(compressed; nthreads=4)
```

### Streaming decompression

```julia
open(InflateZstdStream, "file.zst") do stream
    for line in eachline(stream)
        # ...
    end
end
```

`InflateZstdStream` wraps any readable `IO` object and exposes the
decompressed bytes through the standard `IO` reading interface (`read`,
`readline`, `eof`, ...).

`open(InflateZstdStream, src)` accepts either a path or any readable `IO`.
`close` hands the stream's internal buffers to a shared pool for later streams
to reuse, which is what makes decoding many frames cheap, so prefer the
`do`-block form above — it closes for you, including when the block throws.
Constructing a stream directly is fine too; an unclosed stream is collected
normally, it just rebuilds its buffers from scratch.

Closing a stream does not close an `IO` you passed in — the stream did not open
it, so it does not dispose of it (`own_io = false`, matching `own` in
`Base.fdio`). That makes it safe to run many streams over one open file:

```julia
open(path) do io                       # one file, many independent frames
    while !eof(io)
        frame = open(InflateZstdStream, io) do stream
            read(stream)
        end
        # ...
    end
end
```

Pass `own_io = true` to hand the `IO`'s lifetime to the stream, so that one
`close` disposes of both. Given a *path*, the stream opens the file and always
owns it, since nothing else holds that handle.

Decompression is incremental: compressed bytes are
read from the source one block at a time as output is consumed, and
decompressed output is discarded once it has been read and has aged past
the frame's back-reference window. Memory use is bounded by the frame's
declared window size plus one block (128 KiB), independent of the total
decompressed size.

### Dictionaries

Parse the dictionary once, then pass it to any decompression call:

```julia
dict_bytes = read("my.dict")
dict = parse(ZstdDict, dict_bytes)                        # auto-detects structured vs raw content
# or: parse(ZstdDict, dict_bytes; raw_content=true)       # force raw content mode

data   = inflate_zstd(compressed; dict=dict)
text   = inflate_zstd("file.zst"; dict=dict)

open("file.zst") do io
    data = open(InflateZstdStream, io; dict=dict) do stream
        read(stream)
    end
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

Benchmark scripts comparing the two are in
[`test/performance/`](test/performance/).

## Scope and limitations

- Decompression only; no compression support.
- The maximum supported window size is 2 GiB (the format allows more, but
  larger windows are rejected to bound memory use).
