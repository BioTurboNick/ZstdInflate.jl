# Magic numbers

const ZSTD_MAGIC = 0xFD2FB528                 # (RFC 8878 §3.1.1; identifies the start of a Zstd frame)
const ZSTD_SKIPPABLE_FRAME_MAGIC = 0x184D2A50 # (RFC 8878 §3.1.2; last hex digit may be any value; identifies the start of a skippable frame)
const ZSTD_DICT_MAGIC = 0xEC30A437            # (RFC 8878 §5; identifies the start of a Zstd dictionary)


# Sequence Codes for Lengths and Offsets (RFC 8878 §3.1.1.3.2.1.1)

const MAX_LITERALS_LENGTH = 35
const LITERALS_LENGTH_BASELINE = UInt32[
       0,     1,     2,     3,     4,     5,     6,     7,     8, 
       9,    10,    11,    12,    13,    14,    15,    16,    18,
      20,    22,    24,    28,    32,    40,    48,    64,   128,
     256,   512,  1024,  2048,  4096,  8192, 16384, 32768, 65536
]
const LITERALS_LENGTH_EXTRA_BITS = UInt8[
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     1,  1,  1,  1,  2,  2,  3,  3,  4,  6,  7,  8,  9, 10, 11, 12,
    13, 14, 15, 16
]

const MAX_MATCH_LENGTH = 52
const MATCH_LENGTH_BASELINE = UInt32[
       3,      4,     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,    15, 
       16,    17,    18,    19,    20,    21,    22,    23,    24,    25,    26,    27,    28,    
       29,    30,    31,    32,    33,    34,    35,    37,    39,    41,    43,    47,    51, 
       59,    67,    83,    99,   131,   259,   515,  1027,  2051,  4099,  8195, 16387, 32771, 65539
]
const MATCH_LENGTH_EXTRA_BITS = UInt8[
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     1,  1,  1,  1,  2,  2,  3,  3,  4,  4,  5,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16
]

const MAX_OFFSET_CODE = 31 # Free to choose, minimum recommended is 22, reference uses 31

# Default Distributions (RFC 8878 §3.1.1.3.2.2)

const LITERALS_LENGTH_ACCURACY_LOG = 6
const LITERALS_LENGTH_DEFAULT_DIST = Int16[
     4,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  1,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  2,  1,  1,  1,  1,  1,
    -1, -1, -1, -1
]
const DEFAULT_LL_TABLE = build_fse_table(LITERALS_LENGTH_DEFAULT_DIST, LITERALS_LENGTH_ACCURACY_LOG)

const MATCH_LENGTH_ACCURACY_LOG = 6
const MATCH_LENGTH_DEFAULT_DIST = Int16[
     1,  4,  3,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
    -1, -1, -1, -1, -1
]
const DEFAULT_ML_TABLE = build_fse_table(MATCH_LENGTH_DEFAULT_DIST, MATCH_LENGTH_ACCURACY_LOG)

const OFFSET_ACCURACY_LOG = 5
const OFFSET_DEFAULT_DIST = Int16[
    1,  1,  1,  1,  1,  1,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1
]
const DEFAULT_OFFSET_TABLE = build_fse_table(OFFSET_DEFAULT_DIST, OFFSET_ACCURACY_LOG)


# Initial repeat offsets (RFC 8878 §3.1.1.5)
const INIT_REPEAT_OFFSETS = (1, 4, 8)


# Maximum Huffman table log (RFC 8878 §4.2.1)
const HUFTABLE_LOG_MAX = 11
