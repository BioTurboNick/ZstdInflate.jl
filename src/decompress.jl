# Decode four interleaved Huffman streams into `literals[1:regen_size]`.
# ht::HuffmanTable{L} carries max_bits as a type parameter so that
# safe_n = 57 ÷ L is a compile-time constant; Julia specialises this
# function per distinct L and LLVM unrolls the constant-bound inner loops.
# Called via dynamic dispatch from read_literals — one indirect call per
# outer (refill) iteration, not per symbol.
function _decode_4streams!(data::AbstractVector{UInt8}, ht::HuffmanTable{L},
                            literals::Vector{UInt8}, regen_size::Int) where L
    s1_start = 7
    s2_start = s1_start + Int(_le16(data, 1))
    s3_start = s2_start + Int(_le16(data, 3))
    s4_start = s3_start + Int(_le16(data, 5))
    s4_end = length(data)

    seg_n = (regen_size + 3) >> 2

    # Phase 1: all 4 streams together via SIMD-friendly ReverseBitReader4X.
    # Phase 2: 2-stream pairs chosen by remaining capacity (optimal SIMD utilisation).
    # Phase 3: whichever stream in each pair still has capacity runs single-stream.
    # Phase 4: finish any remaining symbols with single-symbol decode.
    safe_n = 57 ÷ L  # compile-time constant: L is a type parameter
    oi = (1, 1 + seg_n, 1 + 2seg_n, 1 + 3seg_n)  # output indices for each stream
    ends = (seg_n, 2seg_n, 3seg_n, regen_size)
    safeends = (ends[1] - 2safe_n, ends[2] - 2safe_n, ends[3] - 2safe_n, ends[4] - 2safe_n)

    # Phase 1: 4-stream SIMD decode — bits/nbits updated via Vec{4,...} each iteration.
    # safe_n dual-symbol lookups fit in one refill (≥ 57 bits, ≤ max_bits per lookup).
    # safeendX = endX - 2*safe_n: last position where a full batch is safe. Each lookup
    # writes pos and pos+1 unconditionally; the guard ensures pos+1 < endX.
    rb4x = ReverseBitReaderX(
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

    # Phase 2: 2-stream SIMD decode for the two streams with the most remaining work,
    # plus a single-stream 2-symbol loop for the third. The fourth (least remaining —
    # typically the one that caused phase 1 to exit) goes straight to the scalar tail.
    # Sort stream indices by remaining safe-decode capacity; take the top two as the pair.
    r = (safeends[1] - oi[1], safeends[2] - oi[2],
         safeends[3] - oi[3], safeends[4] - oi[4])
    ia, ib, ic, id = sortperm([r[1], r[2], r[3], r[4]], rev=true)

    # Extract all 4 stream readers from rb4x (updated after phase 1).
    s1 = _extract_stream(rb4x, Val(1))
    s2 = _extract_stream(rb4x, Val(2))
    s3 = _extract_stream(rb4x, Val(3))
    s4 = _extract_stream(rb4x, Val(4))
    sv = (s1, s2, s3, s4)

    # Phase 2a: 2X SIMD loop for the top-2 streams (ia, ib).
    rbA = ReverseBitReaderX(sv[ia], sv[ib])
    oi_A = Vec{2, Int}((oi[ia], oi[ib]))
    se_A = Vec{2, Int}((safeends[ia], safeends[ib]))
    while all(oi_A ≤ se_A)
        refill_unchecked!(rbA)
        for _ in 1:safe_n
            nread = decode2x2!(rbA, ht, literals, oi_A)
            oi_A += nread
        end
    end
    ra_ia = _extract_stream(rbA, Val(1))
    ra_ib = _extract_stream(rbA, Val(2))

    # Pair the survivor of phase 2a with ic for phase 2b.
    # At most one of ia/ib can be ≤ safeend after the loop exits.
    ia_alive = oi_A[1] ≤ se_A[1]
    re_2a  = ia_alive ? ra_ia        : ra_ib
    oi_2a  = ia_alive ? Int(oi_A[1]) : Int(oi_A[2])
    se_2a  = ia_alive ? safeends[ia] : safeends[ib]

    # Phase 2b: 2X SIMD loop for (survivor of 2a, ic).
    rbB = ReverseBitReaderX(re_2a, sv[ic])
    oi_B = Vec{2, Int}((oi_2a, oi[ic]))
    se_B = Vec{2, Int}((se_2a, safeends[ic]))
    while all(oi_B ≤ se_B)
        refill_unchecked!(rbB)
        for _ in 1:safe_n
            nread = decode2x2!(rbB, ht, literals, oi_B)
            oi_B += nread
        end
    end
    rb_2b  = _extract_stream(rbB, Val(1))   # survivor-of-2a reader, updated
    rb_ic2 = _extract_stream(rbB, Val(2))   # ic reader, updated

    # Run the survivor of phase 2b single-stream until its safe limit.
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

    # Reconstruct rb1..4 / oi in original stream order.
    # ia: used in 2a slot 1 → rb_2b if ia_alive, else ra_ia
    # ib: used in 2a slot 2 → rb_2b if !ia_alive, else ra_ib
    # ic: used in 2b slot 2 → rb_ic2 (always)
    # id: never used past phase 1 → sv[id]
    final_rb_ia = ia_alive ? rb_2b  : ra_ia
    final_rb_ib = ia_alive ? ra_ib  : rb_2b
    final_rb_ic = rb_ic2
    final_rb_id = sv[id]

    oi_final_ia = ia_alive ? (ie_alive ? oi_2b : Int(oi_B[1])) : Int(oi_A[1])
    oi_final_ib = ia_alive ? Int(oi_A[2]) : (ie_alive ? oi_2b : Int(oi_B[1]))
    oi_final_ic = ie_alive ? Int(oi_B[2]) : oi_2b
    oi_final_id = oi[id]

    rb1 = 1==ia ? final_rb_ia : 1==ib ? final_rb_ib : 1==ic ? final_rb_ic : final_rb_id
    rb2 = 2==ia ? final_rb_ia : 2==ib ? final_rb_ib : 2==ic ? final_rb_ic : final_rb_id
    rb3 = 3==ia ? final_rb_ia : 3==ib ? final_rb_ib : 3==ic ? final_rb_ic : final_rb_id
    rb4 = 4==ia ? final_rb_ia : 4==ib ? final_rb_ib : 4==ic ? final_rb_ic : final_rb_id
    oi = (ia==1 ? oi_final_ia : ib==1 ? oi_final_ib : ic==1 ? oi_final_ic : oi_final_id,
          ia==2 ? oi_final_ia : ib==2 ? oi_final_ib : ic==2 ? oi_final_ic : oi_final_id,
          ia==3 ? oi_final_ia : ib==3 ? oi_final_ib : ic==3 ? oi_final_ic : oi_final_id,
          ia==4 ? oi_final_ia : ib==4 ? oi_final_ib : ic==4 ? oi_final_ic : oi_final_id)

    let p = oi[1]; while p ≤ ends[1]; p += decode1x2_tail!(rb1, ht, literals, p); end; end
    let p = oi[2]; while p ≤ ends[2]; p += decode1x2_tail!(rb2, ht, literals, p); end; end
    let p = oi[3]; while p ≤ ends[3]; p += decode1x2_tail!(rb3, ht, literals, p); end; end
    let p = oi[4]; while p ≤ ends[4]; p += decode1x2_tail!(rb4, ht, literals, p); end; end

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
