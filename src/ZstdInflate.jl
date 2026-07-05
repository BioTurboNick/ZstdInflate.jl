# Pure Julia implementation of Zstandard (Zstd) decompression,
# as specified by RFC 8878.
#
# Reference: https://www.rfc-editor.org/rfc/rfc8878

"""
    ZstdInflate

Pure Julia implementation of decompression of the Zstandard format.

In-memory decompression:

| function | decompresses |
| -------- | ------------ |
| `inflate_zstd(data::Vector{UInt8})` | Zstandard frame |
| `inflate_zstd(filename::AbstractString)` | Zstandard file |

Streaming decompression:

| stream | decompresses |
| ------ | ------------ |
| `InflateZstdStream(stream::IO)` | Zstandard stream |

Reference: [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878)
"""
module ZstdInflate

using SIMD

include("util.jl")
include("xxhash.jl")
include("forwardbitreader.jl")
include("reversebitreader.jl")
include("disttables.jl")
include("huffman.jl")
include("dictionary.jl")
include("constants.jl")
include("decompress.jl")
include("streaming.jl")

export inflate_zstd, InflateZstdStream, ZstdDict

end  # module ZstdInflate
