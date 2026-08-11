# ============================================================
# Streaming interface
#   InflateZstdStream decompresses incrementally: compressed
#   bytes are read from the source IO one block at a time as
#   output is consumed, and decompressed output is discarded
#   once it has been read.
# ============================================================

"""
    InflateZstdStream(io::IO; dict = nothing)

Create a readable stream that incrementally decompresses Zstandard data
from `io`. Compressed bytes are read from `io` one block at a time as
output is consumed, and decompressed output is discarded once it has been
read and has aged past the frame's window, so memory use is bounded by the
frame's declared window size plus one block (128 KiB).

If the data was compressed with a dictionary, pass a `ZstdDict` as `dict`.
Use `Base.parse(ZstdDict, bytes)` to construct one from raw bytes.
"""
mutable struct InflateZstdStream{T <: IO} <: IO
    io          ::T
    dict        ::Union{ZstdDict, Nothing}
    out         ::Vector{UInt8}      # retained output: window history + unconsumed bytes
    read_pos    ::Int                # next unconsumed byte in out
    wpos        ::Int                # next write position in out
    inbuf       ::Vector{UInt8}      # reusable compressed-payload buffer
    hdrbuf      ::Vector{UInt8}      # reusable small buffer for headers/checksums
    # Per-frame decode state (valid while in_frame)
    state       ::DecompressState
    in_frame    ::Bool
    frame_start ::Int                # out-coordinate where the current frame began (≥ 0)
    window_size ::Int
    fcs         ::Int                # declared frame content size; -1 if absent
    frame_len   ::Int                # bytes produced by the current frame so far
    check_flag  ::Bool
    hasher      ::XXH64Stream
    source_done ::Bool               # source IO exhausted at a frame boundary
    mark_pos    ::Int                # out-coordinate of the active mark; -1 if unmarked
    dropped     ::Int                # total bytes ever discarded from out by compaction
    own_io      ::Bool               # whether close(s) also closes s.io
    closed      ::Bool
end

# ------------------------------------------------------------------
# Scratch pooling
#
# A stream's buffers -- the output window, the compressed-payload buffer, the
# header buffer, and the DecompressState with its ~15 vectors and three FSE
# slots -- are all reusable across streams. Decoding many small frames through
# separate streams (one per TIFF strip, say) otherwise rebuilds the lot every
# time, which is the single largest source of allocation in that workload.
#
# `close` returns them here; construction takes them back.
#
# The pool is global and lock-guarded because streams may be on different tasks or threads.
#
# There is no fixed cap. The pool is self-limiting in *count*: a set is only
# created when a stream starts and finds the pool empty, so the number in
# existence cannot exceed the most streams ever live at once, and whatever a
# caller gets back is memory it already paid for at its own high-water mark.
# A fixed cap of 8 was tried and removed: it never bound on a realistic workload
# (peak occupancy decoding a 5478-strip TIFF is 3, at any thread count) while
# above eight live streams it discarded four sets and re-allocated four per
# round.
#
# What is *not* self-limiting is retained size. Julia cannot shrink a Vector's
# capacity, so a set whose output buffer grew for one large frame keeps it: 24
# workers on 10 MB frames leave 12 entries holding 70 MB, and 800 subsequent
# 4 KB frames reclaim none of it. Hence the two mechanisms below — a trim bound
# that follows the caller's concurrency, and idle decay.
# ------------------------------------------------------------------
struct StreamScratch
    out   ::Vector{UInt8}
    inbuf ::Vector{UInt8}
    hdrbuf::Vector{UInt8}
    state ::DecompressState
end

const _SCRATCH_POOL = StreamScratch[]
const _SCRATCH_LOCK = ReentrantLock()

# Sets currently checked out: incremented when one is handed to a stream (pooled
# or freshly built), decremented when one comes back.
#
# A stream that is never closed never returns its set, so this over-counts by the
# number of abandoned streams and the pool merely trims less. Closing is the
# caller's business (see `close`); the cost of not doing it is retained memory,
# not incorrectness.
const _SCRATCH_INFLIGHT = Ref(0)

# High-water mark of `_SCRATCH_INFLIGHT` since the last decay tick, and what the
# trim bound below actually follows.
#
# The instantaneous count is the wrong bound even though it is the obvious one:
# sets come back precisely when a worker finishes, so every trim decision is
# sampled at the trough of the caller's concurrency, not its plateau. With N
# workers cycling through short frames the count dips to 1 between handoffs, the
# bound collapses to 2, and the pool discards sets the other N-1 workers are
# about to ask for. Measured on a 32-strip TIFF at 6 threads that churn cost
# 0.9 GB of extra allocation — the pool was thrashing, not trimming.
#
# Taking the peak over a tick-length window instead means a burst is bounded by
# how wide it actually got, and the decay below is what walks the bound back down
# once the burst is over.
const _SCRATCH_PEAK = Ref(0)

# Idle decay. Created on first return rather than at load: a Timer holds a libuv
# handle, so it cannot live in a `const` (it does not survive precompilation),
# and putting one in `__init__` would impose a periodic wakeup on every process
# that loads this package, including those that never decode. It stops itself
# once the pool is empty, so an idle process stops waking up altogether.
const _SCRATCH_TIMER    = Ref{Union{Timer, Nothing}}(nothing)
const _SCRATCH_TOUCHED  = Ref(false)
const _SCRATCH_INTERVAL = Ref(1.0)   # seconds between decay ticks

# Index of the entry holding the largest output buffer, which is the one worth
# dropping first — size, not count, is what the pool over-retains.
#
# `length` stands in for capacity: Julia offers no portable way to read a
# Vector's capacity (the Memory-based route is 1.11+, and this package supports
# 1.10). Since a stream's output buffer only grows, its length on return is a
# reasonable proxy for how much that set is holding.
function _largest_scratch()
    best, bestlen = firstindex(_SCRATCH_POOL), -1
    for (i, sc) in enumerate(_SCRATCH_POOL)
        n = length(sc.out)
        n > bestlen && ((best, bestlen) = (i, n))
    end
    return best
end

# Drop at most one entry per call, largest first, while above twice the recent
# peak concurrency. One at a time rather than truncating to the bound: a burst
# draining to idle would otherwise throw away everything it is about to want
# again. The floor of 1 keeps a serial caller — whose peak is 1 — from emptying
# the pool it just filled.
function _trim_scratch!()
    length(_SCRATCH_POOL) > 2 * max(_SCRATCH_PEAK[], 1) || return false
    deleteat!(_SCRATCH_POOL, _largest_scratch())
    return true
end

# Caller must hold _SCRATCH_LOCK.
function _start_scratch_decay!()
    _SCRATCH_TIMER[] === nothing || return nothing
    _SCRATCH_TIMER[] = Timer(_SCRATCH_INTERVAL[]; interval = _SCRATCH_INTERVAL[]) do t
        lock(_SCRATCH_LOCK) do
            # Walk the peak back towards what is actually in flight, halving the
            # excess each tick. Halving rather than resetting to the current
            # count keeps a workload that pauses between bursts from having its
            # bound yanked out from under it during the gap.
            live = _SCRATCH_INFLIGHT[]
            _SCRATCH_PEAK[] = live + (_SCRATCH_PEAK[] - live) ÷ 2

            if _SCRATCH_TOUCHED[]
                _SCRATCH_TOUCHED[] = false      # in use; only the bound moves
            elseif !isempty(_SCRATCH_POOL)
                deleteat!(_SCRATCH_POOL, _largest_scratch())
            end
            if isempty(_SCRATCH_POOL)
                close(t)
                _SCRATCH_TIMER[] = nothing
            end
        end
    end
    return nothing
end

_take_scratch!() =
    lock(_SCRATCH_LOCK) do
        _SCRATCH_INFLIGHT[] += 1
        _SCRATCH_PEAK[]     = max(_SCRATCH_PEAK[], _SCRATCH_INFLIGHT[])
        _SCRATCH_TOUCHED[]  = true
        isempty(_SCRATCH_POOL) ? nothing : pop!(_SCRATCH_POOL)
    end

function _give_scratch!(sc::StreamScratch)
    lock(_SCRATCH_LOCK) do
        _SCRATCH_INFLIGHT[] = max(_SCRATCH_INFLIGHT[] - 1, 0)
        _SCRATCH_TOUCHED[]  = true
        push!(_SCRATCH_POOL, sc)
        _trim_scratch!()
        _start_scratch_decay!()
    end
    return nothing
end

"""
    ZstdInflate.empty_scratch_pool!() -> Int

Release every buffer set currently held for reuse by [`InflateZstdStream`], and
return how many were released.

Closed streams hand their internal buffers to a shared pool, which is what makes
decoding many small frames cheap. The pool trims itself towards the number of
streams in flight and decays away once idle, so calling this is optional; it
exists for callers that want the memory back at a particular moment.
"""
function empty_scratch_pool!()
    lock(_SCRATCH_LOCK) do
        n = length(_SCRATCH_POOL)
        empty!(_SCRATCH_POOL)
        # Don't leave a stale burst width behind to inflate the bound afterwards.
        _SCRATCH_PEAK[] = _SCRATCH_INFLIGHT[]
        t = _SCRATCH_TIMER[]
        if t !== nothing
            close(t)
            _SCRATCH_TIMER[] = nothing
        end
        return n
    end
end

"""
    InflateZstdStream(io::IO; dict = nothing, own_io = true)

Create a readable stream that incrementally decompresses Zstandard data
from `io`.

`own_io` controls whether [`close`](@ref) also closes `io`. Pass `own_io =
false` when `io` outlives the stream — decoding a sequence of independent
frames from one open file, for instance — so that `close` releases the
stream's internal buffers for reuse without closing the file underneath.

See `open(InflateZstdStream, io)` for a form that closes the stream for you.
"""
function InflateZstdStream(io::IO; dict::Union{ZstdDict, Nothing} = nothing,
                            own_io::Bool = true)
    sc = _take_scratch!()
    out, inbuf, hdrbuf, state = sc === nothing ?
        (UInt8[], UInt8[], UInt8[], DecompressState()) :
        (sc.out, sc.inbuf, sc.hdrbuf, sc.state)
    empty!(out)
    s = InflateZstdStream{typeof(io)}(io, dict, out, 1, 1, inbuf, hdrbuf,
                                      state, false, 0, 0, -1, 0, false,
                                      XXH64Stream(), false, -1, 0, own_io, false)
    try
        # Parse the first frame header eagerly so structural errors (empty
        # input, bad magic, missing dictionary) surface at construction time.
        eof(io) &&
            throw(ArgumentError("zstd: empty input"))
        _start_frame!(s)
    catch
        # We borrowed a set before we could know the input was bad. Hand it
        # back: otherwise every rejected file leaks both the buffers and an
        # in-flight count, and the count is what bounds the pool — a program
        # that probes files it turns out not to want would push that bound up
        # monotonically until trimming never fires again. `_start_frame!` resets
        # the state on every frame, so a half-started one is fine to reuse.
        _give_scratch!(StreamScratch(s.out, s.inbuf, s.hdrbuf, s.state))
        rethrow()
    end
    return s
end

"""
    open(InflateZstdStream, src; dict = nothing, own_io = true)
    open(f::Function, InflateZstdStream, src; kwargs...)

Open a stream decompressing Zstandard data from `src`, which may be a readable
`IO` or a path to a `.zst` file. The first form is equivalent to the
[`InflateZstdStream`](@ref) constructor; the second applies `f` to the stream,
closes it whether `f` returns or throws, and returns `f`'s value:

```julia
data = open(InflateZstdStream, "file.zst") do s
    read(s)
end
```

Since [`close`](@ref) is what hands the stream's buffers back for reuse by
later streams, this is the cheapest way to decode a sequence of frames. When
`src` is an `IO` that outlives the stream — reading several independent frames
from one open file — pass `own_io = false` so that closing the stream releases
its buffers without closing `src`. When `src` is a path the stream owns the
file it opened, and closing the stream closes it.
"""
Base.open(::Type{InflateZstdStream}, io::IO; kwargs...) =
    InflateZstdStream(io; kwargs...)

function Base.open(::Type{InflateZstdStream}, path::AbstractString; kwargs...)
    io = open(path)
    try
        return InflateZstdStream(io; kwargs...)
    catch
        # The constructor parses the first frame header, so it can throw on a
        # file we just opened; don't leak the handle on the way out.
        close(io)
        rethrow()
    end
end

# The `do` forms come from Base's own `open(f::Function, args...)`, which wraps
# the calls above in exactly the try/finally-close this needs.

# Read exactly n bytes from io into buf (resized to n).
function _read_exact!(io::IO, buf::Vector{UInt8}, n::Int, what::String)
    resize!(buf, n)
    readbytes!(io, buf, n) == n ||
        throw(ArgumentError("zstd: truncated $what"))
    return buf
end

# Read and discard n bytes (skippable frame payload).
function _discard!(io::IO, scratch::Vector{UInt8}, n::Int)
    while n > 0
        k = min(n, 1 << 16)
        resize!(scratch, k)
        nr = readbytes!(io, scratch, k)
        nr > 0 ||
            throw(ArgumentError("zstd: truncated skippable frame (data)"))
        n -= nr
    end
    return
end

# Advance past skippable frames and begin the next zstd frame, parsing its
# header and initialising per-frame state. Returns false if the source ended
# cleanly at a frame boundary.
function _start_frame!(s::InflateZstdStream)
    while true
        eof(s.io) && (s.source_done = true; return false)
        magic = _le32(_read_exact!(s.io, s.hdrbuf, 4, "frame (magic)"), 1)
        if _is_skippable(magic)
            n = Int(Int64(_le32(_read_exact!(s.io, s.hdrbuf, 4, "skippable frame (size)"), 1)))
            _discard!(s.io, s.inbuf, n)
        elseif magic == ZSTD_MAGIC
            break
        else
            throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base = 16))"))
        end
    end

    # Frame Header Descriptor first: it determines how many more header bytes
    # follow (dictionary ID + window descriptor + frame content size).
    fhd = _read_exact!(s.io, s.hdrbuf, 1, "frame (FHD)")[1]
    fcs_size, single_segment_flag, _, dict_id_size, _ =
        _read_frame_header_descriptor(UInt8[fhd], 1)
    rest = dict_id_size + (single_segment_flag ? 0 : 1) + fcs_size

    hdr = Vector{UInt8}(undef, 1 + rest)
    hdr[1] = fhd
    rest > 0 && copyto!(hdr, 2, _read_exact!(s.io, s.hdrbuf, rest, "frame (header)"), 1, rest)

    window_size, frame_content_size, content_checksum_flag, _ = _read_frame_header(hdr, 1, s.dict)
    window_size ≤ STREAM_WINDOW_SIZE_MAX ||
        throw(ArgumentError("zstd: window size $window_size exceeds maximum supported for " *
                            "streaming ($STREAM_WINDOW_SIZE_MAX bytes)"))

    # Reuse the state's scratch across frames rather than rebuilding it; only
    # the fields that carry across blocks within a frame are cleared.
    reset!(s.state, s.dict)
    s.in_frame    = true
    s.frame_start = s.wpos - 1
    s.window_size = window_size
    s.fcs         = frame_content_size
    s.frame_len   = 0

    # Reserve the output up front when the frame declares its size. The reservation is capped
    # by the window plus one block.
    if frame_content_size ≥ 0
        cap  = s.window_size + ZSTD_BLOCKSIZE_MAX
        want = s.frame_start + min(frame_content_size, cap) + WILDCOPY_SLACK
        length(s.out) < want && resize!(s.out, want)
    end
    s.check_flag  = content_checksum_flag
    s.check_flag && xxh_reset!(s.hasher)
    return true
end

# Read and decode the next block of the current frame; on the last block,
# validate the frame content size and content checksum.
function _decode_next_block!(s::InflateZstdStream)
    bh = _read_exact!(s.io, s.hdrbuf, 3, "block header")
    bh1, bh2, bh3 = Int(bh[1]), Int(bh[2]), Int(bh[3])

    last_block = bh1 & 0x01
    block_type = (bh1 >> 1) & 0x03
    block_size = (bh1 >> 3) | (bh2 << 5) | (bh3 << 13)
    block_size ≤ ZSTD_BLOCKSIZE_MAX ||
        throw(ArgumentError("zstd: block size $block_size exceeds maximum (128 KB)"))

    payload_len = block_type == 1 ? 1 : block_size  # RLE payload is a single byte
    payload_len > 0 && _read_exact!(s.io, s.inbuf, payload_len, "block payload")

    # When the frame declares its size, tell the block-decode functions the
    # true remaining bound instead of leaving them to assume "up to one more
    # full block" every time. Without this, read_sequences!'s own reservation
    # math (wpos + ZSTD_BLOCKSIZE_MAX) routinely overshoots what _start_frame!
    # already correctly reserved from fcs, forcing a spurious regrow on
    # nearly every block. It also makes the "block output exceeds declared
    # frame content size" check in _apply_block! actually enforce per-block
    # under streaming, rather than only being caught after the fact once the
    # last block's frame_len mismatch is checked below.
    out_limit = s.fcs ≥ 0 ? s.frame_start + s.fcs : typemax(Int) - WILDCOPY_SLACK

    wpos0 = s.wpos
    s.wpos = _apply_block!(block_type, s.inbuf, 1, block_size, s.state, s.out, s.wpos,
                           false, s.frame_start, out_limit)
    s.frame_len += s.wpos - wpos0
    s.check_flag && xxh_update!(s.hasher, @view s.out[wpos0:s.wpos - 1])

    if last_block != 0
        s.fcs < 0 || s.fcs == s.frame_len ||
            throw(ArgumentError("zstd: decompressed size $(s.frame_len) does not match frame content size $(s.fcs)"))
        if s.check_flag
            stored = _le32(_read_exact!(s.io, s.hdrbuf, 4, "content checksum"), 1)
            computed = UInt32(xxh_finalize(s.hasher) & 0xFFFFFFFF)
            stored == computed ||
                throw(ArgumentError("zstd: content checksum mismatch (stored=0x$(string(stored, base = 16)), computed=0x$(string(computed, base = 16)))"))
        end
        s.in_frame = false
    end
    return
end

# Drop retained output that the consumer has read (or that is pinned by an
# active mark) and that has aged past the window (no future match can
# reference it). Compaction runs only when the droppable prefix is at least
# as large as the retained tail, so the total cost of all copies is O(total
# output).
function _compact!(s::InflateZstdStream)
    consumer_floor = ismarked(s) ? s.mark_pos : s.read_pos
    keep_from = min(consumer_floor, s.wpos - s.window_size)
    drop = keep_from - 1
    (drop ≥ 1 << 16 && drop ≥ s.wpos - keep_from) ||
        return
    copyto!(s.out, 1, s.out, keep_from, s.wpos - keep_from)
    s.wpos        -= drop
    s.read_pos    -= drop
    s.frame_start -= drop
    s.dropped     += drop
    ismarked(s) && (s.mark_pos -= drop)
    if s.frame_start < 0
        # In-frame history older than the window has been dropped; conformant
        # matches can no longer reach the frame start, so the dictionary is
        # unreachable too. Dropping it makes the dict_pos ≥ 1 guard in
        # _run_sequences! reject any (malformed) offset that tries.
        #
        # Rebind rather than `empty!`: the state points straight at the
        # ZstdDict's own content array, which the caller may still be using for
        # other streams. Truncating it in place would empty their dictionary
        # too.
        s.frame_start = 0
        s.state.dict_content = EMPTY_DICT_CONTENT
    end
    return
end

# Decode until at least one unconsumed byte is available or the stream ends.
# Returns true if bytes are available.
function _fill!(s::InflateZstdStream)
    s.closed && _throw_closed()
    while s.read_pos ≥ s.wpos
        if s.in_frame
            _decode_next_block!(s)
            _compact!(s)
        else
            s.source_done && return false
            _start_frame!(s) || return false
        end
    end
    return true
end

Base.eof(s::InflateZstdStream) = !_fill!(s)

function Base.read(s::InflateZstdStream, ::Type{UInt8})
    _fill!(s) || throw(EOFError())
    b = s.out[s.read_pos]
    s.read_pos += 1
    return b
end

function Base.unsafe_read(s::InflateZstdStream, p::Ptr{UInt8}, n::UInt)
    nread = 0
    want  = Int(n)
    while nread < want
        # _fill! returns true only once read_pos < wpos, so k ≥ 1 and this terminates.
        _fill!(s) || throw(EOFError())
        out = s.out
        k   = min(want - nread, s.wpos - s.read_pos)
        GC.@preserve out unsafe_copyto!(p + nread, pointer(out, s.read_pos), k)
        s.read_pos += k
        nread      += k
    end
    return nothing
end

function Base.readbytes!(s::InflateZstdStream, b::AbstractVector{UInt8}, nb = length(b))
    n = 0
    while n < nb && _fill!(s)
        k = min(nb - n, s.wpos - s.read_pos)
        length(b) < n + k && resize!(b, n + k)
        copyto!(b, n + 1, s.out, s.read_pos, k)
        s.read_pos += k
        n += k
    end
    return n
end

# Decoded-but-unconsumed bytes only; does not trigger further decoding.
Base.bytesavailable(s::InflateZstdStream) = s.wpos - s.read_pos

"""
    close(s::InflateZstdStream)

Release `s`'s internal buffers for reuse by later streams, and close the
underlying `io` unless the stream was constructed with `own_io = false`.
Idempotent. Reading from a closed stream throws.

Closing is optional but worth doing when decoding many frames: an unclosed
stream is collected normally, it just rebuilds its buffers from scratch instead
of taking a set from the pool, and it leaves the pool's trim bound believing one
more stream is still running than really is. `open(InflateZstdStream, io) do s
... end` closes for you. Use [`ZstdInflate.empty_scratch_pool!`](@ref) to
release the pool outright.
"""
function Base.close(s::InflateZstdStream)
    s.closed && return nothing
    s.closed = true
    _give_scratch!(StreamScratch(s.out, s.inbuf, s.hdrbuf, s.state))
    # Drop our references so a use-after-close reads nothing that now belongs
    # to another stream; `closed` turns it into a clear error either way.
    s.out    = UInt8[]
    s.inbuf  = UInt8[]
    s.hdrbuf = UInt8[]
    s.read_pos = 1
    s.wpos     = 1
    s.in_frame = false
    s.own_io && close(s.io)
    return nothing
end

Base.isopen(s::InflateZstdStream) = !s.closed && isopen(s.io)

@noinline _throw_closed() = throw(ArgumentError("zstd: stream is closed"))

# Absolute count of decompressed bytes consumed so far, stable across
# internal buffer compaction.
Base.position(s::InflateZstdStream) = s.dropped + s.read_pos - 1

Base.ismarked(s::InflateZstdStream) = s.mark_pos != -1

"""
    mark(s::InflateZstdStream) -> Int64

Mark the current position in the decompressed output. A later `reset(s)`
rewinds to this position. While a mark is held, decompressed output back to
the mark is retained even past the frame's window, so an outstanding mark
disables the usual bounded-memory compaction until it is cleared by `reset`
or `unmark`.
"""
function Base.mark(s::InflateZstdStream)
    s.mark_pos = s.read_pos
    return position(s)
end

"""
    unmark(s::InflateZstdStream) -> Bool

Remove any mark on `s` without resetting to it. Returns whether a mark was
present.
"""
function Base.unmark(s::InflateZstdStream)
    had = ismarked(s)
    s.mark_pos = -1
    return had
end

"""
    reset(s::InflateZstdStream) -> Int64

Rewind `s` to its last `mark`ed position and remove the mark. Throws
`ArgumentError` if `s` is not marked.
"""
function Base.reset(s::InflateZstdStream)
    ismarked(s) || throw(ArgumentError("InflateZstdStream not marked"))
    s.read_pos = s.mark_pos
    s.mark_pos = -1
    return position(s)
end

"""
    seekstart(s::InflateZstdStream) -> s

Rewind `s` to the beginning of the decompressed output by seeking the
underlying `io` back to its start and restarting decoding from there.
Requires that `io` itself supports `seekstart`.
"""
function Base.seekstart(s::InflateZstdStream)
    seekstart(s.io)
    eof(s.io) &&
        throw(ArgumentError("zstd: empty input"))
    resize!(s.out, 0)
    s.read_pos    = 1
    s.wpos        = 1
    s.dropped     = 0
    s.mark_pos    = -1
    s.in_frame    = false
    s.source_done = false
    _start_frame!(s)
    return s
end
