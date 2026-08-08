# ============================================================
# Huffman decode table
#   Reference: RFC 8878 §4.2.1.3
# ============================================================

#=
The Huffman decode table is a flat array of 2^L entries, where `L` is the maximum code length of any symbol.
The index into the table is the next `L` read from the bitstream. Each entry contains either one or two symbols,
the number of bits of the stream they take up, and the number of symbols present. If nsymbols == 1, the second
symbol entry is invalid. The design allows us to opportunistically read a second symbol in the same read.
=#

struct HuffmanTableEntry{L} # TODO: The type param may not be needed, test it
    symbols::NTuple{2, UInt8} # Second symbol is valid only if nbits_total > nbits_sym1
    stream_nbits::UInt8       # [0:L]
    nsymbols::UInt8           # 0, 1 or 2 (0 should only be present during table construction; 0 means invalid entry) TODO: Check if this is true

    function HuffmanTableEntry{L}(symbols::NTuple{2, UInt8}, stream_nbits::UInt8, nsymbols::UInt8) where L
        stream_nbits ≤ L ||
            throw(ArgumentError("zstd: Huffman table entry stream_nbits $stream_nbits exceeds max_bits ($L)"))
        0 ≤ nsymbols ≤ 2 ||
            throw(ArgumentError("zstd: Huffman table entry nsymbols must be 0, 1 or 2"))
        new{L}(symbols, stream_nbits, nsymbols)
    end
    function HuffmanTableEntry{L}() where L
        new{L}((0x00, 0x00), 0x00, 0x00)
    end
end

struct HuffmanTable{L, T <: AbstractVector{HuffmanTableEntry{L}}}
    decode_table::T
end

Base.@propagate_inbounds Base.getindex(ht::HuffmanTable, args...) = getindex(ht.decode_table, args...)

# Build a dual-symbol Huffman decode table from a weight array.
function build_huffman_table!(weights::Vector{UInt8}, max_bits::Int; kwargs...)
    max_bits > 0 ||
        throw(ArgumentError("zstd: all-zero Huffman weights"))
    max_bits ≤ HUFTABLE_LOG_MAX ||
        throw(ArgumentError("zstd: Huffman table log $max_bits exceeds maximum ($HUFTABLE_LOG_MAX)"))

    table_size = 1 << max_bits
    v = fill(HuffmanTableEntry{max_bits}(), table_size)
    # Keep a function barrier between the runtime value and function working on the concrete type
    return build_huffman_table!(v, weights; kwargs...)
end

function build_huffman_table!(decode_table::AbstractVector{HuffmanTableEntry{L}}, weights::Vector{UInt8};
                               scratch_buffers::Union{Nothing, Tuple{AbstractVector{Int}, AbstractVector{Int}, AbstractVector{UInt8}}} = nothing) where L
    # Pass 1: Populate single-symbol entries for all symbols
    if isa(scratch_buffers, Tuple{AbstractVector{Int}, AbstractVector{Int}, AbstractVector{UInt8}})
        rank_count = resize!(fill!(scratch_buffers[1], 0x00), L)
        next_rank_start = resize!(fill!(scratch_buffers[2], 0x00), L)
        nbits_sym1s = resize!(scratch_buffers[3], length(decode_table))
    else
        rank_count = zeros(Int, L)
        next_rank_start = zeros(Int, L)
        nbits_sym1s = Vector{UInt8}(undef, length(decode_table))
    end
    for w ∈ weights
        w > 0 || continue
        w ≤ L ||
            throw(ArgumentError("zstd: Huffman weight $w exceeds table log ($L)"))
        rank_count[w] += 1
    end
    # Running total of the entries claimed by each rank. Using for loop to avoid
    # capturing and boxing `rank_count`.
    acc = 0
    for w in 1:(L - 1)
        acc += rank_count[w] * (1 << (w - 1))
        next_rank_start[w + 1] = acc
    end
    for (i, w) ∈ enumerate(weights)
        w > 0 || continue
        sym = UInt8(i - 1)
        nbits_sym1 = UInt8(L - w + 1)
        entry = HuffmanTableEntry{L}((sym, 0x00), nbits_sym1, 0x01)
        start = next_rank_start[w]
        num_entries = 1 << (w - 1)
        decode_table[start .+ (1:num_entries)] .= Ref(entry)
        next_rank_start[w] += num_entries
    end

    # Pass 2: Add second symbols where there is room in the entry (i.e., nbits1 + nbits2 ≤ max_bits)
    # nbits_sym1s snapshots each entry's pass-1 stream_nbits before this pass
    # starts mutating decode_table -- some entries get looked up (as `j`) both
    # before and after their own turn in this loop, so a live read from
    # decode_table itself would sometimes see an already-mutated 2-symbol
    # entry's combined nbits instead of its original single-symbol value.
    for i in eachindex(decode_table)
        nbits_sym1s[i] = decode_table[i].stream_nbits
    end
    for (i, entry) ∈ enumerate(decode_table)
        code = i - 1
        nbits_remaining = L - entry.stream_nbits
        nbits_remaining > 0 || continue
        code2 = (code << entry.stream_nbits) & (length(decode_table) - 1)
        j = code2 + 1
        nbits_sym2 = Int(nbits_sym1s[j])
        nbits_sym2 ≤ nbits_remaining || continue
        entry2 = decode_table[j]
        stream_nbits = UInt8(entry.stream_nbits + nbits_sym2)
        decode_table[i] = HuffmanTableEntry{L}((entry.symbols[1], entry2.symbols[1]), stream_nbits, 0x02)
    end

    return HuffmanTable{L, typeof(decode_table)}(decode_table)
end

# ============================================================
# Huffman tree loading
#   Reference: RFC 8878 §4.2.1
# ============================================================

function read_huffman_description(data::AbstractVector{UInt8};
                                   scratch_buffers::Union{Nothing, Tuple{AbstractVector{UInt8}, AbstractVector{Int}, AbstractVector{Int}, AbstractVector{UInt8}}} = nothing,
                                   rb_scratch::Union{Nothing, ReverseBitReader} = nothing)
    length(data) ≥ 1 ||
        throw(ArgumentError("zstd: truncated Huffman table description"))
    headerByte = Int(data[1]) # RFC 8878 §4.2.1.1
    weights = scratch_buffers !== nothing ? scratch_buffers[1] : UInt8[]
    is_fse_encoded = headerByte < 128
    if is_fse_encoded
        nbytes = headerByte
        length(data) ≥ nbytes + 1 ||
            throw(ArgumentError("zstd: truncated Huffman table description"))
        br = ForwardBitReader(@view data[2:nbytes + 1])
        _, table_log = _read_fse_weights!(weights, br, nbytes, rb_scratch)
    else
        nsyms = headerByte - 127
        nbytes = (nsyms + 1) >> 1
        length(data) ≥ nbytes + 1 ||
            throw(ArgumentError("zstd: truncated Huffman table description"))
        weightdata = @view data[2:nbytes + 1]
        _, table_log = _read_direct_weights!(weights, weightdata, nsyms)
    end
    scratch_buffers !== nothing && (scratch_buffers = scratch_buffers[2:4])
    ht = build_huffman_table!(weights, table_log; scratch_buffers)
    return ht, nbytes + 1
end

function _read_direct_weights!(weights::Vector{UInt8}, data::AbstractVector{UInt8}, nsyms::Int)
    nbytes = (nsyms + 1) >> 1
    resize!(weights, nsyms + 1)
    for i in 1:nbytes
        b = data[i]
        j = (i - 1) * 2 + 1
        lowbits, weights[j] = splitbyte(b)
        j + 1 ≤ nsyms &&
            (weights[j + 1] = lowbits)
    end
    last_w, table_log = _infer_last_weight(weights, nsyms)
    weights[end] = last_w
    return weights, table_log
end

# RFC 8878 §4.2.1.2
function _read_fse_weights!(weights::Vector{UInt8}, br::ForwardBitReader, byte_limit::Int,
                             rb_scratch::Union{Nothing, ReverseBitReader} = nothing)
    al, dist = read_fse_dist!(br, HUFTABLE_LOG_MAX)
    t = build_fse_table(dist, al)

    pos_after = byte_pos(br)
    n_remain = byte_limit - pos_after + 1
    n_remain > 0 ||
        throw(ArgumentError("zstd: no data for Huffman weight FSE stream"))

    view = @view br.data[pos_after:pos_after + n_remain - 1]
    rb = rb_scratch !== nothing ? reinit!(rb_scratch, view) : ReverseBitReader(view)

    state1 = dist_table_init!(rb, t)
    state2 = dist_table_init!(rb, t)

    empty!(weights)
    while true
        sym1 = dist_table_peek(t, state1)
        state1 = _fse_update_unchecked(rb, t, state1)
        push!(weights, UInt8(sym1))
        if _rbr_overflowed(rb) || length(weights) ≥ 255
            push!(weights, UInt8(dist_table_peek(t, state2)))
            break
        end

        sym2 = dist_table_peek(t, state2)
        state2 = _fse_update_unchecked(rb, t, state2)
        push!(weights, UInt8(sym2))
        if _rbr_overflowed(rb) || length(weights) ≥ 255
            push!(weights, UInt8(dist_table_peek(t, state1)))
            break
        end
    end

    last_w, table_log = _infer_last_weight(weights)
    push!(weights, UInt8(last_w))

    return weights, table_log
end

# `n` is the number of weights to sum. It is an explicit argument because the
# direct-weight path reserves a trailing slot in `weights` for the weight this
# function infers; that slot must not be summed.
function _infer_last_weight(weights::AbstractVector{UInt8}, n::Int = length(weights))
    total = 0
    @inbounds for i in 1:n
        w = Int(weights[i])
        w > 0 && (total += 1 << (w - 1))
    end
    total > 0 || return (1, 1)  # single symbol edge case
    table_log = _flog2(total) + 1
    p = UInt64(1) << table_log
    p > total || (table_log += 1; p <<= 1)
    rest = Int(p - total)
    rest > 0 && (rest & (rest - 1)) == 0 ||
        throw(ArgumentError("zstd: invalid Huffman weight sum"))
    last_w = _flog2(rest) + 1
    return last_w, table_log
end


# ============================================================
# Huffman stream decoding
#   Reference: RFC 8878 §4.2.2
# ============================================================

# @noinline throw helper keeps the cold error path out of _decode_4streams!'s
# codegen (guarded by an instruction/call-count regression test).
@noinline _throw_invalid_stream_sizes() =
    throw(ArgumentError("zstd: invalid literals stream sizes"))

# Permutation of (1,2,3,4) descending by r[i], i.e. r[i[1]] ≥ r[i[2]] ≥ r[i[3]] ≥ r[i[4]].
# Equivalent to sortperm(collect(r); rev=true) for exactly 4 elements, via the
# standard optimal 5-comparator sorting network -- no array, no allocation.
# Only feeds a load-balancing heuristic (which stream pair phase 2A tackles
# first), not decode correctness, so exact tie-break behavior doesn't need to
# match sortperm's.
@inline function _sortperm4_desc(r::NTuple{4, Int})
    i1, i2, i3, i4 = 1, 2, 3, 4
    r[i1] < r[i2] && ((i1, i2) = (i2, i1))
    r[i3] < r[i4] && ((i3, i4) = (i4, i3))
    r[i1] < r[i3] && ((i1, i3) = (i3, i1))
    r[i2] < r[i4] && ((i2, i4) = (i4, i2))
    r[i2] < r[i3] && ((i2, i3) = (i3, i2))
    return (i1, i2, i3, i4)
end

# Decode the four Huffman streams stored in `data` using the lookup table `ht` and store
# the result in `literals`. This code is tuned to promote LLVM SIMD instructions; changes
# in it or the functions it calls could break this. Use caution.
function _decode_4streams!(data::AbstractVector{UInt8}, ht::HuffmanTable{L},
                            literals::Vector{UInt8}, regen_size::Int,
                            scratch::Huffman4StreamScratch) where L
    # Read stream-start indexes from the 6-byte jump table (RFC 8878 §3.1.1.3.1.6).
    # The caller guarantees length(data) ≥ 10 (jump table + four non-empty
    # streams); the stream boundaries themselves are attacker-controlled and
    # must be validated before use.
    s1_start = 7
    s2_start = s1_start + Int(_le16(data, 1))
    s3_start = s2_start + Int(_le16(data, 3))
    s4_start = s3_start + Int(_le16(data, 5))
    s4_end = length(data)
    (s1_start < s2_start < s3_start < s4_start ≤ s4_end) ||
        _throw_invalid_stream_sizes()

    seg_n = (regen_size + 3) >> 2
    safe_n = 57 ÷ L
    oi = (1, 1 + seg_n, 1 + 2seg_n, 1 + 3seg_n)
    ends = (seg_n, 2seg_n, 3seg_n, regen_size)
    safeends = (ends[1] - 2safe_n, ends[2] - 2safe_n, ends[3] - 2safe_n, ends[4] - 2safe_n)

    # Phase 1: SIMD parallel processing of the four streams until at least one is exhausted (within safe window)
    rb4x = reinit!(scratch.rb4x,
        @view(data[s1_start:s2_start-1]),
        @view(data[s2_start:s3_start-1]),
        @view(data[s3_start:s4_start-1]),
        @view(data[s4_start:s4_end]),
    )
    oi_vec = Vec{4, Int}(oi)
    safeends_vec = Vec{4, Int}(safeends)
    while all(oi_vec ≤ safeends_vec)
        refill_unchecked!(rb4x)
        for _ in 1:safe_n
            nread = decode4x2!(rb4x, ht, literals, oi_vec)
            oi_vec += nread
        end
    end
    oi = Tuple(oi_vec)  # spill Vec back to scalar for remaining phases

    # Phase 2A: SIMD parallel processing of the top pair of streams with the most work remaining
    r = (safeends[1] - oi[1], safeends[2] - oi[2],
         safeends[3] - oi[3], safeends[4] - oi[4])
    ia, ib, ic, id = _sortperm4_desc(r)

    s1 = _extract_stream!(scratch.s1, rb4x, Val(1))
    s2 = _extract_stream!(scratch.s2, rb4x, Val(2))
    s3 = _extract_stream!(scratch.s3, rb4x, Val(3))
    s4 = _extract_stream!(scratch.s4, rb4x, Val(4))
    sv = (s1, s2, s3, s4)

    rbA = reinit!(scratch.rbA, sv[ia], sv[ib])
    oi_A = Vec{2, Int}((oi[ia], oi[ib]))
    se_A = Vec{2, Int}((safeends[ia], safeends[ib]))
    while all(oi_A ≤ se_A)
        refill_unchecked!(rbA)
        for _ in 1:safe_n
            nread = decode2x2!(rbA, ht, literals, oi_A)
            oi_A += nread
        end
    end
    ra_ia = _extract_stream!(scratch.ra_ia, rbA, Val(1))
    ra_ib = _extract_stream!(scratch.ra_ib, rbA, Val(2))

    # Phase 2B: SIMD parallel processing of the survivor with the last unexhausted stream
    ia_alive = oi_A[1] ≤ se_A[1]
    re_2a  = ia_alive ? ra_ia        : ra_ib
    oi_2a  = ia_alive ? Int(oi_A[1]) : Int(oi_A[2])
    se_2a  = ia_alive ? safeends[ia] : safeends[ib]

    rbB = reinit!(scratch.rbB, re_2a, sv[ic])
    oi_B = Vec{2, Int}((oi_2a, oi[ic]))
    se_B = Vec{2, Int}((se_2a, safeends[ic]))
    while all(oi_B ≤ se_B)
        refill_unchecked!(rbB)
        for _ in 1:safe_n
            nread = decode2x2!(rbB, ht, literals, oi_B)
            oi_B += nread
        end
    end
    rb_2b  = _extract_stream!(scratch.rb_2b, rbB, Val(1))    # survivor-of-2a reader, updated
    rb_ic2 = _extract_stream!(scratch.rb_ic2, rbB, Val(2))   # ic reader, updated

    # Phase 2C: Process remaining unexhausted stream
    ie_alive = oi_B[1] ≤ se_B[1]
    re_2b  = ie_alive ? rb_2b  : rb_ic2
    oi_2b  = ie_alive ? Int(oi_B[1]) : Int(oi_B[2])
    se_2b  = ie_alive ? se_2a  : safeends[ic]
    while oi_2b ≤ se_2b
        refill!(re_2b)
        for _ in 1:safe_n
            nread = decode1x2!(re_2b, ht, literals, oi_2b)
            oi_2b += nread
        end
    end

    # Phase 3: Process any remaining tails of all four streams
    rbs = (ia_alive ? rb_2b  : ra_ia, ia_alive ? ra_ib  : rb_2b, rb_ic2, sv[id])
    oi = (
        ia_alive ? (ie_alive ? oi_2b : Int(oi_B[1])) : Int(oi_A[1]),
        ia_alive ? Int(oi_A[2]) : (ie_alive ? oi_2b : Int(oi_B[1])),
        ie_alive ? Int(oi_B[2]) : oi_2b,
        oi[id]
    )
    perm = (ia, ib, ic, id)

    let p = oi[1]; while p ≤ ends[perm[1]]; p += decode1x2_tail!(rbs[1], ht, literals, p); end; end
    let p = oi[2]; while p ≤ ends[perm[2]]; p += decode1x2_tail!(rbs[2], ht, literals, p); end; end
    let p = oi[3]; while p ≤ ends[perm[3]]; p += decode1x2_tail!(rbs[3], ht, literals, p); end; end
    let p = oi[4]; while p ≤ ends[perm[4]]; p += decode1x2_tail!(rbs[4], ht, literals, p); end; end

    return
end

# Read 1-2 symbols from 4 streams and return the number of symbols read
# Always writes 2 symbols even if only the first is valid; up to the caller to provide room
@inline function decode4x2!(rb::ReverseBitReaderX{4}, ht::HuffmanTable{L}, out::Vector{UInt8}, oi::Vec{4, Int}) where L
    i = peek(rb, Val(L))
    @inbounds entry = (
        ht[i[1] % Int + 1],
        ht[i[2] % Int + 1],
        ht[i[3] % Int + 1],
        ht[i[4] % Int + 1]
    )
    GC.@preserve out begin
        unsafe_store!.(Ptr{NTuple{2, UInt8}}.(pointer.(Ref(out), Tuple(oi))), htol.(getfield.(entry, :symbols)))
    end
    nbits_consumed = (Int(entry[1].stream_nbits), Int(entry[2].stream_nbits),
                      Int(entry[3].stream_nbits), Int(entry[4].stream_nbits))
    skip(rb, nbits_consumed)
    return Vec{4, Int64}((Int64(entry[1].nsymbols), Int64(entry[2].nsymbols),
                          Int64(entry[3].nsymbols), Int64(entry[4].nsymbols)))
end

# Read 1-2 symbols from 2 streams and return the number of symbols read
# Always writes 2 symbols even if only the first is valid; up to the caller to provide room
@inline function decode2x2!(rb::ReverseBitReaderX{2}, ht::HuffmanTable{L}, out::Vector{UInt8}, oi::Vec{2, Int}) where L
    i = peek(rb, Val(L))
    @inbounds entry = (
        ht[i[1] % Int + 1],
        ht[i[2] % Int + 1]
    )
    GC.@preserve out begin
        unsafe_store!.(Ptr{NTuple{2, UInt8}}.(pointer.(Ref(out), Tuple(oi))), htol.(getfield.(entry, :symbols)))
    end
    nbits_consumed = (Int(entry[1].stream_nbits), Int(entry[2].stream_nbits))
    skip(rb, nbits_consumed)
    return Vec{2, Int64}((Int64(entry[1].nsymbols), Int64(entry[2].nsymbols)))
end

# Read 1-2 symbols from 1 stream and return the number of symbols read
# Always writes 2 symbols even if only the first is valid; up to the caller to provide room
@inline function decode1x2!(rb::ReverseBitReader, ht::HuffmanTable{L}, out::Vector{UInt8}, o::Int) where L
    i = peek(rb, Val(L))
    @inbounds entry = ht[i[1] % Int + 1]
    GC.@preserve out begin
        unsafe_store!(Ptr{NTuple{2, UInt8}}(pointer(out, o)), htol(entry.symbols))
    end
    nbits_consumed = Int(entry.stream_nbits)
    skip(rb, nbits_consumed)
    return Int64(entry.nsymbols)
end

# Read 1-2 symbols from 1 stream and return the number of symbols read
# Does not write second symbol if it is not present
@inline function decode1x2_tail!(rb::ReverseBitReader, ht::HuffmanTable{L}, out::Vector{UInt8}, o::Int) where L
    rb.nbits ≥ L || refill!(rb)
    i = peek(rb, Val(L))
    entry = @inbounds ht.decode_table[i + 1]
    nbits_consumed = Int(entry.stream_nbits)
    @inbounds out[o] = entry.symbols[1]
    skip(rb, nbits_consumed)
    if entry.nsymbols == 2
        @inbounds out[o + 1] = entry.symbols[2]
    end
    return entry.nsymbols
end
