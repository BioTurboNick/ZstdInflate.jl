"""
    InflateZstdStream(io::IO; dict = nothing, nthreads = Threads.nthreads())

Create a readable stream that decompresses Zstandard data from `io`.
The stream reads all compressed data at construction time; subsequent
reads deliver decompressed bytes.

If the data was compressed with a dictionary, pass a `ZstdDict` as `dict`.
Use `Base.parse(ZstdDict, bytes)` to construct one from raw bytes.
"""
mutable struct InflateZstdStream <: IO
    buf::Vector{UInt8}
    pos::Int
end

function InflateZstdStream(io::IO; dict::Union{ZstdDict,Nothing} = nothing, nthreads::Int = Threads.nthreads())
    compressed = read(io)
    decompressed = inflate_zstd(compressed; dict=dict, nthreads=nthreads)
    InflateZstdStream(decompressed, 1)
end

Base.eof(s::InflateZstdStream) = s.pos > length(s.buf)

function Base.read(s::InflateZstdStream, ::Type{UInt8})
    s.pos ≤ length(s.buf) || throw(EOFError())
    b = s.buf[s.pos]
    s.pos += 1
    return b
end

function Base.readbytes!(s::InflateZstdStream, b::AbstractVector{UInt8}, nb=length(b))
    available = length(s.buf) - s.pos + 1
    n = min(nb, available)
    n ≤ 0 && return 0
    length(b) < n && resize!(b, n)
    copyto!(b, 1, s.buf, s.pos, n)
    s.pos += n
    return n
end

Base.bytesavailable(s::InflateZstdStream) = max(0, length(s.buf) - s.pos + 1)
