# RFC 8878 §3.1.1
function _decompress_frame!(data::Vector{UInt8}, pos::Int, out::Vector{UInt8},
                            dict::Union{ZstdDict, Nothing} = nothing)
    # Magic number (4 bytes, little-endian)
    magic = _read_magic(data, pos)
    magic == ZSTD_MAGIC ||
        throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base = 16))"))
    pos += 4

    # Frame Header Descriptor (FHD)
    window_size, frame_content_size, content_checksum_flag, pos = _read_frame_header(data, pos, dict)

    # Enforce a maximum window size to prevent memory exhaustion (2 GiB); spec maximum is (1 << 41) + 7 * (1 << 38)
    window_size ≤ Int64(1) << 31 ||
        throw(ArgumentError("zstd: window size $window_size exceeds maximum supported (2 GiB)"))

    state = dict !== nothing ? DecompressState(dict) : DecompressState()
    frame_start = length(out)
    # When FCS is known, resize to the exact frame size upfront so that all
    # per-block writes go directly into pre-allocated space — no per-block
    # resize! or ml_vals total scan needed.  When FCS is unknown, rely on the
    # caller's one-time sizehint! and fall back to per-block resize!.
    preallocated = frame_content_size ≥ 0
    if preallocated
        resize!(out, frame_start + frame_content_size + 15)
    end
    wpos = frame_start + 1

    # Decode blocks
    while true
        length(data) ≥ pos + 2 ||
            throw(ArgumentError("zstd: truncated block header"))
        # Block header is 3 bytes
        bh1, bh2, bh3 = Int(data[pos]), Int(data[pos + 1]), Int(data[pos + 2])
        pos += 3

        last_block  = bh1 & 0x01
        block_type  = (bh1 >> 1) & 0x03
        # block_size for compressed/raw/RLE: 21-bit field in remaining 21 bits
        block_size  = (bh1 >> 3) | (bh2 << 5) | (bh3 << 13)
        block_size ≤ ZSTD_BLOCKSIZE_MAX ||
            throw(ArgumentError("zstd: block size $block_size exceeds maximum (128 KB)"))

        is_raw = block_type == 0
        is_rle = block_type == 1
        is_compressed = block_type == 2

        if is_raw
            pos + block_size - 1 ≤ length(data) ||
                throw(ArgumentError("zstd: truncated raw block"))
            preallocated ||
                resize!(out, wpos - 1 + block_size)
            GC.@preserve out data Base.memcpy(pointer(out, wpos), pointer(data, pos), block_size)
            wpos += block_size
            pos += block_size
        elseif is_rle
            preallocated ||
                resize!(out, wpos - 1 + block_size)
            fill!(view(out, wpos:wpos + block_size - 1), data[pos])
            wpos += block_size
            pos += 1
        elseif is_compressed
            wpos = _decompress_block!(data, pos, block_size, state, out, wpos, preallocated)
            pos += block_size
        else
            throw(ArgumentError("zstd: unsupported block type (reserved)"))
        end

        last_block != 0 && break
    end

    # Trim slack bytes added for wildcopy16 over-writes
    resize!(out, wpos - 1)

    frame_len = wpos - 1 - frame_start

    # Validate Frame Content Size (RFC 8878 §3.1.1.1.4)
    frame_content_size < 0 || frame_content_size == frame_len ||
        throw(ArgumentError("zstd: decompressed size $frame_len does not match frame content size $frame_content_size"))

    # Content checksum (optional 4 bytes)
    if content_checksum_flag
        length(data) ≥ pos + 3 ||
            throw(ArgumentError("zstd: truncated content checksum"))
        stored = _le32(data, pos)
        pos += 4
        computed = UInt32(xxhash64(@view(out[frame_start+1:end])) & 0xFFFFFFFF)
        stored == computed ||
            throw(ArgumentError("zstd: content checksum mismatch (stored=0x$(string(stored,base=16)), computed=0x$(string(computed,base=16)))"))
    end

    return pos
end

function _decompress_block!(data::Vector{UInt8}, pos::Int, block_size::Int, state::DecompressState,
                           out::Vector{UInt8}, wpos::Int, preallocated::Bool)
    limit = pos + block_size - 1
    literals, lit_consumed = read_literals(data, pos, state)
    seq_pos = pos + lit_consumed
    nextwpos = wpos
    if seq_pos ≤ limit
        nextwpos = read_sequences!(data, seq_pos, limit, state, literals, out, wpos, preallocated)
    end
    if nextwpos == wpos
        # No sequences section: all literals + literals has 15 bytes of slack
        lit_len = length(literals) - 15
        if !preallocated
            resize!(out, wpos - 1 + lit_len + 15)
        end
        GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, 1), lit_len)
        return wpos + lit_len
    else
        return nextwpos
    end
end

# Reference: RFC 8878 §3.1.1.3.1
function read_literals(data::Vector{UInt8}, pos::Int, state::DecompressState)
    br = ForwardBitReader(@view data[pos:end])
    litblock_type = read(br, 2)
    size_format = peek(br, 2)

    is_raw = litblock_type == 0
    is_rle = litblock_type == 1
    is_treeless = litblock_type == 3

    if is_raw || is_rle
        size_format_nbits = iseven(size_format) ? 1 :
                                                  2
        size_nbits, header_nbytes = iseven(size_format) ? ( 5, 1) :
                                    size_format == 1    ? (12, 2) :
                                                          (20, 3)
        skip(br, size_format_nbits)
        regen_size = Int(read(br, size_nbits))
        compressed_size = is_raw ? regen_size :
                                   1

        literals = state.literals_buf
        resize!(literals, regen_size + LITERALS_WILDCOPY_SLACK)

        block_start = pos + header_nbytes

        if is_raw
            copyto!(literals, 1, data, block_start, regen_size)
        else
            literals[1:regen_size] .= data[block_start]
        end

        return literals, header_nbytes + compressed_size
    else # Compressed (2) or Treeless (3)
        size_nbits, header_nbytes = size_format < 2  ? (10, 3) :
                                    size_format == 2 ? (14, 4) :
                                                       (18, 5)
        num_streams = size_format == 0 ? 1 :
                                         4
        
        skip(br, 2)
        regen_size = Int(read(br, size_nbits))
        compressed_size = Int(read(br, size_nbits))

        payload_start = pos + header_nbytes
        payload_end = payload_start + compressed_size - 1

        if is_treeless
            state.huffman !== nothing ||
                throw(ArgumentError("zstd: treeless literals but no prior Huffman table"))
            ht = state.huffman
            huf_start = payload_start
        else # Compressed
            ht, hdr_len = read_huffman_description((@view data[payload_start:payload_end]); scratch_buffers = (state.huf_weights, state.huf_rank_count, state.huf_rank_start))
            state.huffman = ht
            huf_start = payload_start + hdr_len
        end

        literals = state.literals_buf
        resize!(literals, regen_size + LITERALS_WILDCOPY_SLACK)

        if num_streams == 1
            stream_len = payload_end - huf_start + 1
            rb = ReverseBitReader(@view data[huf_start:huf_start + stream_len - 1])
            let p = 1
                while p ≤ regen_size
                    p += decode1x2_tail!(rb, ht, literals, p)
                end
            end
        else
            _decode_4streams!((@view data[huf_start:payload_end]), ht, literals, regen_size)
        end

        resize!(literals, regen_size + LITERALS_WILDCOPY_SLACK)  # trim dual-symbol slack

        return literals, header_nbytes + compressed_size
    end
end

# Reference: RFC 8878 §3.1.1.3.2
function read_sequences!(data::Vector{UInt8}, pos::Int, limit::Int,
                         state::DecompressState, literals::Vector{UInt8},
                         out::Vector{UInt8}, wpos::Int, preallocated::Bool)
    pos > limit && return wpos

    # RFC 8878 §3.1.1.3.2
    b0 = Int(data[pos]);  pos += 1
    local num_seqs::Int
    if b0 < 128
        num_seqs = b0
    elseif b0 < 255
        num_seqs = ((b0 - 128) << 8) | Int(data[pos]);  pos += 1
    else
        num_seqs = Int(data[pos]) + (Int(data[pos+1]) << 8) + 0x7F00;  pos += 2
    end

    num_seqs == 0 && return wpos

    # Symbol Compression Modes byte (RFC 8878 §3.1.1.3.2.1)
    modes_byte = Int(data[pos]);  pos += 1
    modes_byte & 0x03 == 0 || throw(ArgumentError("zstd: reserved bits set in Symbol_Compression_Modes, must be zero"))
    ll_mode = (modes_byte >> 6) & 0x03
    of_mode = (modes_byte >> 4) & 0x03
    ml_mode = (modes_byte >> 2) & 0x03

    br = ForwardBitReader(@view data[pos:limit])
    ll_tab = read_distribution_table!(br, DEFAULT_LITERALS_LENGTH_TABLE, state.ll_tab, ll_mode, MAX_LITERALS_LENGTH, 9,
                             state.ll_slot, state)
    of_tab = read_distribution_table!(br, DEFAULT_OFFSET_TABLE, state.of_tab, of_mode, MAX_OFFSET_CODE, 8,
                             state.of_slot, state)
    ml_tab = read_distribution_table!(br, DEFAULT_MATCH_LENGTH_TABLE, state.ml_tab, ml_mode, MAX_MATCH_LENGTH, 9,
                             state.ml_slot, state)
    state.ll_tab = ll_tab
    state.of_tab = of_tab
    state.ml_tab = ml_tab

    # The bitstream for sequences is a reverse bitstream starting right after
    # the distribution table descriptions.
    seq_start = byte_pos(br)
    seq_len = limit - seq_start + 1
    seq_len > 0 || throw(ArgumentError("zstd: no data for sequences bitstream"))

    rb = ReverseBitReader(@view data[seq_start:seq_start + seq_len - 1])

    # Init distribution table states
    ll_state = dist_table_init!(rb, ll_tab)
    of_state = dist_table_init!(rb, of_tab)
    ml_state = dist_table_init!(rb, ml_tab)

    resize!(state.ll_vals, num_seqs)
    resize!(state.ml_vals, num_seqs)
    resize!(state.of_vals, num_seqs)
    ll_vals = state.ll_vals
    ml_vals = state.ml_vals
    of_vals = state.of_vals

    for i in 1:num_seqs
        # Peek symbols from all three states (no bits consumed)
        ll_code = dist_table_peek(ll_tab, ll_state)
        ml_code = dist_table_peek(ml_tab, ml_state)
        of_code = dist_table_peek(of_tab, of_state)
        of_code ≤ MAX_OFFSET_CODE ||
            throw(ArgumentError("zstd: offset code $of_code exceeds maximum supported value, $MAX_OFFSET_CODE"))

        of_n  = of_code
        ml_n  = Int(@inbounds MATCH_LENGTH_EXTRA_BITS[ml_code + 1])
        ll_n  = Int(@inbounds LITERALS_LENGTH_EXTRA_BITS[ll_code + 1])

        # State-transition widths (skip on last sequence)
        update = i < num_seqs
        ll_nb = update ? _dist_table_nb_bits(ll_tab, ll_state) : 0
        ml_nb = update ? _dist_table_nb_bits(ml_tab, ml_state) : 0
        of_nb = update ? _dist_table_nb_bits(of_tab, of_state) : 0

        total_n = of_n + ml_n + ll_n + ll_nb + ml_nb + of_nb

        # Read in batches for optimal ILP
        if total_n ≤ 57
            # Fast path: a single refill guarantees ≥ 57 bits available.
            rb.nbits < total_n && refill!(rb)
            rb.nbits ≥ total_n || throw(ArgumentError("zstd: unexpected end of sequence bitstream"))

            # Cumulative bit offsets into the frozen snapshot
            c_ml  = of_n
            c_ll  = c_ml + ml_n
            c_llb = c_ll + ll_n
            c_mlb = c_llb + ll_nb
            c_ofb = c_mlb + ml_nb

            bits     = rb.bits
            of_extra = Int(ifelse(of_n  > 0, _shr(bits,              64 - of_n ), 0))
            ml_extra = Int(ifelse(ml_n  > 0, _shr(_shl(bits, c_ml ), 64 - ml_n ), 0))
            ll_extra = Int(ifelse(ll_n  > 0, _shr(_shl(bits, c_ll ), 64 - ll_n ), 0))
            ll_bits  = Int(ifelse(ll_nb > 0, _shr(_shl(bits, c_llb), 64 - ll_nb), 0))
            ml_bits  = Int(ifelse(ml_nb > 0, _shr(_shl(bits, c_mlb), 64 - ml_nb), 0))
            of_bits  = Int(ifelse(of_nb > 0, _shr(_shl(bits, c_ofb), 64 - of_nb), 0))

            # Single bulk consume
            rb.bits   = bits << total_n
            rb.nbits -= total_n

        else
            # Slow path: large offset (of_code > ~20); sequential reads.
            of_extra = Int(read(rb, of_n ))
            ml_extra = Int(read(rb, ml_n))
            ll_extra = Int(read(rb, ll_n))
            ll_bits  = Int(read(rb, ll_nb))
            ml_bits  = Int(read(rb, ml_nb))
            of_bits  = Int(read(rb, of_nb))
        end

        of_val64 = (Int64(1) << of_code) + of_extra
        of_val64 ≤ typemax(Int) || throw(ArgumentError("zstd: offset value $of_val64 exceeds addressable range"))
        of_vals[i] = Int(of_val64)
        ml_vals[i] = Int(@inbounds MATCH_LENGTH_BASELINE[ml_code + 1]) + ml_extra
        ll_vals[i] = Int(@inbounds LITERALS_LENGTH_BASELINE[ll_code + 1]) + ll_extra

        if update
            ll_state = _dist_table_baseline(ll_tab, ll_state) + ll_bits
            ml_state = _dist_table_baseline(ml_tab, ml_state) + ml_bits
            of_state = _dist_table_baseline(of_tab, of_state) + of_bits
        end
    end

    return execute_sequences!(ll_vals, ml_vals, of_vals, literals, state, out, wpos, preallocated)
end

# Execute decoded sequences to produce output bytes.
# Writes starting at wpos in out; returns the next write position.
# When preallocated=true the caller has already resize!'d out to the exact frame size,
# so the total scan and per-block resize! can be skipped entirely.
#
# FUTURE OPTIMISATION — fuse sequence decode and execute:
#
# Currently read_sequences! decodes all sequences into three Int arrays
# (ll_vals, ml_vals, of_vals) and then execute_sequences! replays them.
# This is two passes: the sequence data is written to and read back from
# memory.  Fusing the FSE decode loop directly into the execute loop would
# halve that memory traffic and allow the compiler to interleave FSE state
# updates with literal copy and match copy, improving ILP.
#
# FUTURE OPTIMISATION — wildcopy for short non-overlapping matches:
#
# The non-overlapping match path (offset ≥ ml) calls Base.memcpy (a C FFI
# call) for every match.  For short matches (≤ ~32 bytes), the ccall overhead
# dominates the actual data movement cost.  Replacing short non-overlapping
# match copies with _wildcopy16! (same pattern as literal scatter) would
# eliminate that overhead.  Requires extending the +15 slack on `out` to cover
# over-writes at the match destination, and capping wildcopy to matches that
# cannot overlap (offset ≥ 16 would be sufficient for a 16-byte chunk size).

# Reference: RFC 8878 §3.1.1.4
function execute_sequences!(
        ll_vals::Vector{Int}, ml_vals::Vector{Int}, of_vals::Vector{Int},
        literals::Vector{UInt8}, state::DecompressState,
        out::Vector{UInt8}, wpos::Int, preallocated::Bool)

    n = length(ll_vals)

    if !preallocated
        # Pre-size output: every literal byte + every match byte will be written exactly once.
        # literals has 15 bytes of slack; subtract them so total reflects actual content.
        # Add 15 bytes of slack at the end for _wildcopy16! over-writes.
        total = length(literals) - 15
        @inbounds for i in 1:n
            total += ml_vals[i]
        end
        resize!(out, wpos - 1 + total + 15)
    end

    lit_pos = 1

    @inbounds for i in 1:n
        ll = ll_vals[i]
        ml = ml_vals[i]
        of = of_vals[i]

        # Copy ll literal bytes.  out and literals are distinct arrays so no overlap is possible.
        if ll > 0
            GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, lit_pos), ll)
            wpos    += ll
            lit_pos += ll
        end

        # Determine actual offset from repeat-offset table
        # of is the raw Offset_Value; 1/2/3 are repeat codes, ≥4 is a new offset.
        rep = state.rep
        local offset::Int
        if of > 3
            offset = of - 3
            state.rep = (offset, rep[1], rep[2])
        elseif ll > 0
            # Normal repeat-offset rules
            if of == 1
                offset = rep[1]
                # no rep update
            elseif of == 2
                offset = rep[2]
                state.rep = (rep[2], rep[1], rep[3])
            else  # of == 3
                offset = rep[3]
                state.rep = (rep[3], rep[1], rep[2])
            end
        else
            # LL==0: repeat-offset references shift up by 1
            if of == 1
                offset = rep[2]
                state.rep = (rep[2], rep[1], rep[3])
            elseif of == 2
                offset = rep[3]
                state.rep = (rep[3], rep[1], rep[2])
            else  # of == 3
                offset = rep[1] - 1
                offset > 0 || throw(ArgumentError("zstd: repeat offset - 1 is zero"))
                state.rep = (offset, rep[1], rep[2])
            end
        end

        # Copy match of length ml from offset back in output.
        # The match may reach into the dictionary content prefix.
        # wpos - 1 is the logical end of written output; match_pos is 1-indexed into out.
        dict     = state.dict_content
        dict_len = length(dict)
        match_pos = wpos - offset   # = (wpos - 1) - offset + 1
        if match_pos < 1
            # Offset reaches into dictionary content
            dict_pos = dict_len + match_pos      # 1-indexed into dict
            dict_pos ≥ 1 || throw(ArgumentError("zstd: match offset $offset beyond dictionary and output"))
            for _ in 1:ml
                if dict_pos ≤ dict_len
                    out[wpos] = dict[dict_pos]
                    wpos     += 1
                    dict_pos += 1
                else
                    out[wpos]  = out[match_pos]
                    wpos      += 1
                    match_pos += 1
                end
            end
        else
            if offset ≥ ml
                # Non-overlapping match.  For short copies, _wildcopy16! avoids the
                # libc memcpy FFI call; for larger copies memcpy wins (wider SIMD).
                if ml ≤ 64
                    GC.@preserve out _wildcopy16!(pointer(out, wpos), pointer(out, match_pos), ml)
                else
                    GC.@preserve out Base.memcpy(pointer(out, wpos), pointer(out, match_pos), ml)
                end
            elseif offset == 1
                # Single-byte repeat: fill
                @inbounds fill!(view(out, wpos:wpos+ml-1), out[match_pos])
            else
                # Overlapping repeat pattern: copy base pattern once, then
                # keep doubling by copying already-written output.  Each
                # memcpy is non-overlapping (filled bytes precede dest).
                GC.@preserve out Base.memcpy(pointer(out, wpos), pointer(out, match_pos), offset)
                filled = offset
                while filled < ml
                    to_copy = min(filled, ml - filled)
                    GC.@preserve out Base.memcpy(pointer(out, wpos + filled), pointer(out, wpos), to_copy)
                    filled += to_copy
                end
            end
            wpos += ml
        end
    end

    # Remaining literals after last sequence.
    # Use regen_size (stored in literals length - 15 slack) to get true count.
    rem = length(literals) - 15 - lit_pos + 1
    if rem > 0
        GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, lit_pos), rem)
        wpos += rem
    end
    return wpos
end


# ============================================================
# Huffman tree loading
#   Reference: RFC 8878 §4.2.1
# ============================================================

function read_huffman_description(data::AbstractVector{UInt8}; scratch_buffers::Union{Nothing, Tuple{AbstractVector{UInt8}, AbstractVector{Int}, AbstractVector{Int}}} = nothing)
    headerByte = Int(data[1]) # RFC 8878 §4.2.1.1
    weights = scratch_buffers !== nothing ? scratch_buffers[1] : UInt8[]
    is_fse_encoded = headerByte < 128
    if is_fse_encoded
        nbytes = headerByte
        br = ForwardBitReader(@view data[2:nbytes + 1])
        _, table_log = _read_fse_weights!(weights, br, nbytes)
    else
        nsyms = headerByte - 127
        nbytes = (nsyms + 1) >> 1
        weightdata = @view data[2:nbytes + 1]
        _, table_log = _read_direct_weights!(weights, weightdata, nsyms)
    end
    scratch_buffers !== nothing && (scratch_buffers = scratch_buffers[2:3])
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
    last_w, table_log = _infer_last_weight(weights)
    weights[end] = last_w
    return weights, table_log
end

# RFC 8878 §4.2.1.2
function _read_fse_weights!(weights::Vector{UInt8}, br::ForwardBitReader, byte_limit::Int)
    al, dist = read_fse_dist!(br, HUFTABLE_LOG_MAX)
    t = build_fse_table(dist, al)

    pos_after = byte_pos(br)
    n_remain = byte_limit - pos_after + 1
    n_remain > 0 ||
        throw(ArgumentError("zstd: no data for Huffman weight FSE stream"))

    rb = ReverseBitReader(@view br.data[pos_after:pos_after + n_remain - 1])

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

function _infer_last_weight(weights::Vector{UInt8})
    total = Int(sum(w -> UInt64(1) << (Int(w) - 1), weights))
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

# Decode the four Huffman streams stored in `data` using the lookup table `ht` and store
# the result in `literals`. This code is tuned to promote LLVM SIMD instructions; changes
# in it or the functions it calls could break this. Use caution.
function _decode_4streams!(data::AbstractVector{UInt8}, ht::HuffmanTable{L},
                            literals::Vector{UInt8}, regen_size::Int) where L
    # Read stream-start indexes
    s1_start = 7
    s2_start = s1_start + Int(_le16(data, 1))
    s3_start = s2_start + Int(_le16(data, 3))
    s4_start = s3_start + Int(_le16(data, 5))
    s4_end = length(data)

    seg_n = (regen_size + 3) >> 2
    safe_n = 57 ÷ L
    oi = (1, 1 + seg_n, 1 + 2seg_n, 1 + 3seg_n)
    ends = (seg_n, 2seg_n, 3seg_n, regen_size)
    safeends = (ends[1] - 2safe_n, ends[2] - 2safe_n, ends[3] - 2safe_n, ends[4] - 2safe_n)

    # Phase 1: SIMD parallel processing of the four streams until at least one is exhausted (within safe window)
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

    # Phase 2A: SIMD parallel processing of the top pair of streams with the most work remaining
    r = (safeends[1] - oi[1], safeends[2] - oi[2],
         safeends[3] - oi[3], safeends[4] - oi[4])
    ia, ib, ic, id = sortperm([r[1], r[2], r[3], r[4]], rev=true)

    s1 = _extract_stream(rb4x, Val(1))
    s2 = _extract_stream(rb4x, Val(2))
    s3 = _extract_stream(rb4x, Val(3))
    s4 = _extract_stream(rb4x, Val(4))
    sv = (s1, s2, s3, s4)

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

    # Phase 2B: SIMD parallel processing of the survivor with the last unexhausted stream
    ia_alive = oi_A[1] ≤ se_A[1]
    re_2a  = ia_alive ? ra_ia        : ra_ib
    oi_2a  = ia_alive ? Int(oi_A[1]) : Int(oi_A[2])
    se_2a  = ia_alive ? safeends[ia] : safeends[ib]

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
