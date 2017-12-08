/*
        qmx_codec.hpp (modified by Matthias Petri)

        Taken from COMPRESS_INTEGER_QMX_IMPROVED.CPP
        ---------------------------------
        Copyright (c) 2014-2017 Andrew Trotman
        Released under the 2-clause BSD license
   (See:https://en.wikipedia.org/wiki/BSD_licenses)

        A version of BinPacking where we pack into a 128-bit SSE register the
   following:
                256  0-bit words
                128  1-bit words
                 64	 2-bit words
                 40  3-bit words
                 32  4-bit words
                 24  5-bit words
                 20  6-bit words
                 16  8-bit words
                 12 10-bit words
                  8 16-bit words
                  4 32-bit words
                or pack into two 128-bit words (i.e. 256 bits) the following:
                 36  7-bit words
                 28  9-bit words
                 20 12-bit words
                 12 21-bit words

        This gives us 15 possible combinations.  The combinaton is stored in the
   top 4 bits of a selector byte.  The
        bottom 4-bits of the selector store a run-length (the number of such
   sequences seen in a row.

        The 128-bit (or 256-bit) packed binary values are stored first.  Then we
   store the selectors,  Finally,
        stored variable byte encoded, is a pointer to the start of the selector
   (from the end of the sequence).

        This way, all reads and writes are 128-bit word aligned, except
   addressing the selector (and the pointer
        the selector).  These reads are byte aligned.

        Note:  There is curly 1 unused encoding (i.e. 16 unused selecvtor
   values).  These might in the future be
        used for encoding exceptions, much as PForDelta does.
*/

#include <array>
#include <vector>

#include <emmintrin.h>
#include <smmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace QMX {
namespace constants {
    const uint32_t WASTAGE = 512;

    struct type_and_integers {
        size_t type;
        size_t integers;
    };

    // clang-format off
static const type_and_integers table[] = {
        {0, 256}, // size_in_bits == 0;
        {1, 128}, // size_in_bits == 1;
        {2, 64},  // size_in_bits == 2;
        {3, 40},  // size_in_bits == 3;
        {4, 32},  // size_in_bits == 4;
        {5, 24},  // size_in_bits == 5;
        {6, 20},  // size_in_bits == 6;
        {7, 36},  // size_in_bits == 7; 256-bits
        {8, 16},  // size_in_bits == 8;
        {9, 28},  // size_in_bits == 9; 256-bits
        {10, 12}, // size_in_bits == 10;
        {0, 0},
        {11, 20}, // size_in_bits == 12;
        {0, 0},
        {0, 0},
        {0, 0},
        {12, 8},  // size_in_bits == 16;
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {13, 12}, // size_in_bits == 21; 256-bits
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {14, 4},  // size_in_bits == 32;
        };

    alignas(16) static uint32_t static_mask_21[] = {0x1fffff, 0x1fffff, 0x1fffff, 0x1fffff};
    alignas(16) static uint32_t static_mask_12[] = {0xfff, 0xfff, 0xfff, 0xfff};
    alignas(16) static uint32_t static_mask_10[] = {0x3ff, 0x3ff, 0x3ff, 0x3ff};
    alignas(16) static uint32_t static_mask_9[]  = {0x1ff, 0x1ff, 0x1ff, 0x1ff};
    alignas(16) static uint32_t static_mask_7[]  = {0x7f, 0x7f, 0x7f, 0x7f};
    alignas(16) static uint32_t static_mask_6[]  = {0x3f, 0x3f, 0x3f, 0x3f};
    alignas(16) static uint32_t static_mask_5[]  = {0x1f, 0x1f, 0x1f, 0x1f};
    alignas(16) static uint32_t static_mask_4[]  = {0x0f, 0x0f, 0x0f, 0x0f};
    alignas(16) static uint32_t static_mask_3[]  = {0x07, 0x07, 0x07, 0x07};
    alignas(16) static uint32_t static_mask_2[]  = {0x03, 0x03, 0x03, 0x03};
    alignas(16) static uint32_t static_mask_1[]  = {0x01, 0x01, 0x01, 0x01};
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"

namespace util {

    template <class T> T maximum(T a, T b) { return a > b ? a : b; }

    template <class T> T maximum(T a, T b, T c, T d)
    {
        return maximum(maximum(a, b),maximum(c, d));
    }

    // How many QMX bits are needed to store and integer of the given value
    static uint8_t bits_needed_for(uint32_t value)
    {
        if (value == 0x01)
            return 0;
        else if (value <= 0x01)
            return 1;
        else if (value <= 0x03)
            return 2;
        else if (value <= 0x07)
            return 3;
        else if (value <= 0x0F)
            return 4;
        else if (value <= 0x1F)
            return 5;
        else if (value <= 0x3F)
            return 6;
        else if (value <= 0x7F)
            return 7;
        else if (value <= 0xFF)
            return 8;
        else if (value <= 0x1FF)
            return 9;
        else if (value <= 0x3FF)
            return 10;
        else if (value <= 0xFFF)
            return 12;
        else if (value <= 0xFFFF)
            return 16;
        else if (value <= 0x1FFFFF)
            return 21;
        else
            return 32;
    }

    // clang-format on
}

template <uint32_t block_size> struct codec {
    static_assert(block_size % 8 == 0, "Block size must be multiple of 8");
    std::vector<uint8_t> length_buffer;
    std::vector<uint32_t> full_length_buffer;
    codec()
    {
        length_buffer.resize(block_size + constants::WASTAGE);
        full_length_buffer.resize(block_size + constants::WASTAGE);
    }

    void write_out(uint8_t** buffer, uint32_t* src, uint32_t raw_count, uint32_t size_in_bits, uint8_t** length_buffer)
    {
        uint32_t cur, batch;
        uint8_t* dest = *buffer;
        uint8_t* key_store = *length_buffer;
        uint32_t sequence_buffer[4];
        uint32_t instance, value;
        uint8_t type;
        uint32_t count;

        type = constants::table[size_in_bits].type;
        count = (raw_count + constants::table[size_in_bits].integers - 1) / constants::table[size_in_bits].integers;

        // 0-pad if there aren't enough integers in the src buffer.
        auto flb = full_length_buffer.data();
        if (constants::table[size_in_bits].type != 0 && count * constants::table[size_in_bits].integers != raw_count) {
            // must 0-pad to prevent read overflow in input buffer
            std::copy(src, src + raw_count, flb);
            std::fill(flb + raw_count, flb + (count * constants::table[size_in_bits].integers), 0);
            src = flb;
        }
        uint32_t* end = src + raw_count;
        while (count > 0) {
            batch = count > 16 ? 16 : count;
            *key_store++ = (type << 4) | (~(batch - 1) & 0x0F);
            count -= batch;
            for (cur = 0; cur < batch; cur++) {
                switch (size_in_bits) {
                case 0: // 0 bits per integer (i.e. a long sequence of zeros)
                    // we don't need to store a 4 byte integer because its
                    // implicit
                    src += 256;
                    break;
                case 1: // 1 bit per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 128; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 1);

                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 128;
                    break;
                case 2: // 2 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 64; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 2);

                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 64;
                    break;
                case 3: // 3 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 40; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 3);

                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 40;
                    break;
                case 4: // 4 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 32; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 4);

                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 32;
                    break;
                case 5: // 5 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 24; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 5);

                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 24;
                    break;
                case 6: // 6 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 20; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 6);
                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 20;
                    break;
                case 7: // 7 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 20; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 7);
                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;

                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 16; value < 20; value++)
                        sequence_buffer[value & 0x03] |= src[value] >> 4;
                    for (value = 20; value < 36; value++)
                        sequence_buffer[value & 0x03] |= src[value] << (((value - 20) / 4) * 7 + 3);
                    memcpy(dest, sequence_buffer, 16);

                    dest += 16;
                    src += 36; // 36 in a double 128-bit word
                    break;
                case 8: // 8 bits per integer
                    for (instance = 0; instance < 16 && src < end; instance++)
                        *dest++ = (uint8_t)*src++;
                    break;
                case 9: // 9 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 16; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 9);
                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;

                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 12; value < 16; value++)
                        sequence_buffer[value & 0x03] |= src[value] >> 5;
                    for (value = 16; value < 28; value++)
                        sequence_buffer[value & 0x03] |= src[value] << (((value - 16) / 4) * 9 + 4);
                    memcpy(dest, sequence_buffer, 16);

                    dest += 16;
                    src += 28; // 28 in a double 128-bit word
                    break;
                case 10: // 10 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 12; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 10);

                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;
                    src += 12;
                    break;
                case 12: // 12 bit integers
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 12; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 12);
                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;

                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 8; value < 12; value++)
                        sequence_buffer[value & 0x03] |= src[value] >> 8;
                    for (value = 12; value < 20; value++)
                        sequence_buffer[value & 0x03] |= src[value] << (((value - 12) / 4) * 12 + 8);
                    memcpy(dest, sequence_buffer, 16);

                    dest += 16;
                    src += 20; // 20 in a double 128-bit word
                    break;
                case 16: // 16 bits per integer
                    for (instance = 0; instance < 8 && src < end; instance++) {
                        *(uint16_t*)dest = (uint16_t)*src++;
                        dest += 2;
                    }
                    break;
                case 21: // 21 bits per integer
                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 0; value < 8; value++)
                        sequence_buffer[value & 0x03] |= src[value] << ((value / 4) * 21);
                    memcpy(dest, sequence_buffer, 16);
                    dest += 16;

                    memset(sequence_buffer, 0, sizeof(sequence_buffer));
                    for (value = 4; value < 8; value++)
                        sequence_buffer[value & 0x03] |= src[value] >> 11;
                    for (value = 8; value < 12; value++)
                        sequence_buffer[value & 0x03] |= src[value] << (((value - 8) / 4) * 21 + 11);
                    memcpy(dest, sequence_buffer, 16);

                    dest += 16;
                    src += 12; // 12 in a double 128-bit word
                    break;
                case 32: // 32 bits per integer
                    for (instance = 0; instance < 4 && src < end; instance++) {
                        *(uint32_t*)dest = (uint32_t)*src++;
                        dest += 4;
                    }
                    break;
                }
            }
        }
        *buffer = dest;
        *length_buffer = key_store;
    }

    size_t encode(void* dest_as_void, const uint32_t* src)
    {
        uint32_t* dest32 = static_cast<uint32_t*>(dest_as_void);
        uint8_t *cur_len, *dest = (uint8_t *)dest32, *keys;
        uint32_t rlen, bits, new_needed, wastage;
        uint32_t block, largest;
        uint8_t* len_buf = length_buffer.data();
        const uint32_t* cur;

        // (1) Get the lengths of the integers
        cur_len = len_buf;
        for (cur = src; cur < src + block_size; cur += 8) {
            *(cur_len) = util::bits_needed_for(*cur);
            *(cur_len + 1) = util::bits_needed_for(*(cur + 1));
            *(cur_len + 2) = util::bits_needed_for(*(cur + 2));
            *(cur_len + 3) = util::bits_needed_for(*(cur + 3));
            *(cur_len + 4) = util::bits_needed_for(*(cur + 4));
            *(cur_len + 5) = util::bits_needed_for(*(cur + 5));
            *(cur_len + 6) = util::bits_needed_for(*(cur + 6));
            *(cur_len + 7) = util::bits_needed_for(*(cur + 7));
            cur_len += 8;
        }

        // (2) Add 0 length integers on the end to allow for overflow
        for (wastage = 0; wastage < constants::WASTAGE; wastage++) {
            *cur_len++ = 0;
        }

        /*
           (3) Process the lengths.  To maximise SSE throughput we need
           each write to be 128-bit (4*32-bit) aligned and therefore
           we need each compress "block" to be the same size where a
           compressed "block" is a set of four encoded integers
           starting on a 4-integer boundary.
        */
        for (uint8_t* cl = len_buf; cl < len_buf + block_size + 4; cl += 4) {
            *cl = *(cl + 1) = *(cl + 2) = *(cl + 3) = util::maximum(*cl, *(cl + 1), *(cl + 2), *(cl + 3));
        }

        /*
           This code makes sure we can do aligned reads, promoting to
           larger integers if necessary
        */
        cur_len = len_buf;
        while (cur_len < len_buf + block_size) {
            /*
               If there are fewer than 16 values remaining and they all fit
               into 8-bits then its smaller than storing stripes

               If there are fewer than 8 values remaining and they all fit
               into 16-bits then its smaller than storing stripes

               If there are fewer than 4 values remaining and they all fit
               into 32-bits then its smaller than storing stripes
            */
            if (block_size - (cur_len - len_buf) < 4) {
                largest = 0;
                for (block = 0; block < 8; block++)
                    largest = util::maximum((uint8_t)largest, *(cur_len + block));
                if (largest <= 8)
                    for (block = 0; block < 8; block++)
                        *(cur_len + block) = 8;
                else if (largest <= 16)
                    for (block = 0; block < 8; block++)
                        *(cur_len + block) = 16;
                else if (largest <= 32)
                    for (block = 0; block < 8; block++)
                        *(cur_len + block) = 32;
            } else if (block_size - (cur_len - len_buf) < 8) {
                largest = 0;
                for (block = 0; block < 8; block++)
                    largest = util::maximum((uint8_t)largest, *(cur_len + block));
                if (largest <= 8)
                    for (block = 0; block < 8; block++)
                        *(cur_len + block) = 8;
                else if (largest <= 8)
                    for (block = 0; block < 8; block++)
                        *(cur_len + block) = 16;
            } else if (block_size - (cur_len - len_buf) < 16) {
                largest = 0;
                for (block = 0; block < 16; block++)
                    largest = util::maximum((uint8_t)largest, *(cur_len + block));
                if (largest <= 8)
                    for (block = 0; block < 16; block++)
                        *(cur_len + block) = 8;
            }

            // Otherwise we have the standard rules for a block
            switch (*cur_len) {
            case 0:
                for (block = 0; block < 256; block += 4)
                    if (*(cur_len + block) > 0)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 1; // promote
                if (*cur_len == 0) {
                    for (block = 0; block < 256; block++)
                        cur_len[block] = 0;
                    cur_len += 256;
                }
                break;
            case 1:
                for (block = 0; block < 128; block += 4)
                    if (*(cur_len + block) > 1)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 2; // promote
                if (*cur_len == 1) {
                    for (block = 0; block < 128; block++)
                        cur_len[block] = 1;
                    cur_len += 128;
                }
                break;
            case 2:
                for (block = 0; block < 64; block += 4)
                    if (*(cur_len + block) > 2)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 3; // promote
                if (*cur_len == 2) {
                    for (block = 0; block < 64; block++)
                        cur_len[block] = 2;
                    cur_len += 64;
                }
                break;
            case 3:
                for (block = 0; block < 40; block += 4)
                    if (*(cur_len + block) > 3)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 4; // promote
                if (*cur_len == 3) {
                    for (block = 0; block < 40; block++)
                        cur_len[block] = 3;
                    cur_len += 40;
                }
                break;
            case 4:
                for (block = 0; block < 32; block += 4)
                    if (*(cur_len + block) > 4)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 5; // promote
                if (*cur_len == 4) {
                    for (block = 0; block < 32; block++)
                        cur_len[block] = 4;
                    cur_len += 32;
                }
                break;
            case 5:
                for (block = 0; block < 24; block += 4)
                    if (*(cur_len + block) > 5)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 6; // promote
                if (*cur_len == 5) {
                    for (block = 0; block < 24; block++)
                        cur_len[block] = 5;
                    cur_len += 24;
                }
                break;
            case 6:
                for (block = 0; block < 20; block += 4)
                    if (*(cur_len + block) > 6)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 7; // promote
                if (*cur_len == 6) {
                    for (block = 0; block < 20; block++)
                        cur_len[block] = 6;
                    cur_len += 20;
                }
                break;
            case 7:
                for (block = 0; block < 36; block += 4) // 36 in a double 128-bit word
                    if (*(cur_len + block) > 7)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 8; // promote
                if (*cur_len == 7) {
                    for (block = 0; block < 36; block++)
                        cur_len[block] = 7;
                    cur_len += 36;
                }
                break;
            case 8:
                for (block = 0; block < 16; block += 4)
                    if (*(cur_len + block) > 8)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 9; // promote
                if (*cur_len == 8) {
                    for (block = 0; block < 16; block++)
                        cur_len[block] = 8;
                    cur_len += 16;
                }
                break;
            case 9:
                for (block = 0; block < 28; block += 4) // 28 in a double 128-bit word
                    if (*(cur_len + block) > 9)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 10; // promote
                if (*cur_len == 9) {
                    for (block = 0; block < 28; block++)
                        cur_len[block] = 9;
                    cur_len += 28;
                }
                break;
            case 10:
                for (block = 0; block < 12; block += 4)
                    if (*(cur_len + block) > 10)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 12; // promote
                if (*cur_len == 10) {
                    for (block = 0; block < 12; block++)
                        cur_len[block] = 10;
                    cur_len += 12;
                }
                break;
            case 12:
                for (block = 0; block < 20; block += 4) // 20 in a double 128-bit word
                    if (*(cur_len + block) > 12)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 16; // promote
                if (*cur_len == 12) {
                    for (block = 0; block < 20; block++)
                        cur_len[block] = 12;
                    cur_len += 20;
                }
                break;
            case 16:
                for (block = 0; block < 8; block += 4)
                    if (*(cur_len + block) > 16)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 21; // promote
                if (*cur_len == 16) {
                    for (block = 0; block < 8; block++)
                        cur_len[block] = 16;
                    cur_len += 8;
                }
                break;
            case 21:
                for (block = 0; block < 12; block += 4) // 12 in a double 128-bit word
                    if (*(cur_len + block) > 21)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 32; // promote
                if (*cur_len == 21) {
                    for (block = 0; block < 12; block++)
                        cur_len[block] = 21;
                    cur_len += 12;
                }
                break;
            case 32:
                for (block = 0; block < 4; block += 4)
                    if (*(cur_len + block) > 32)
                        *cur_len = *(cur_len + 1) = *(cur_len + 2) = *(cur_len + 3) = 64; // promote
                if (*cur_len == 32) {
                    for (block = 0; block < 4; block++)
                        cur_len[block] = 32;
                    cur_len += 4;
                }
                break;
            default:
                exit(printf("Selecting on a non whole power of 2\n"));
                break;
            }
        }

        // We can now compress based on the lengths in len_buf
        rlen = 1;
        bits = len_buf[0];
        // we're going to re-use the len_buf because it can't overlap
        keys = len_buf;
        for (cur = (uint32_t*)src + 1; cur < src + block_size; cur++) {
            new_needed = len_buf[cur - src];
            if (new_needed == bits) {
                rlen++;
            } else {
                write_out(&dest, (uint32_t*)cur - rlen, rlen, bits, &keys);
                bits = new_needed;
                rlen = 1;
            }
        }
        write_out(&dest, (uint32_t*)cur - rlen, rlen, bits, &keys);

        // Copy the lengths to the end, backwards
        uint8_t* from = len_buf + (keys - len_buf) - 1;
        uint8_t* to = dest;
        for (uint32_t pos = 0; pos < keys - len_buf; pos++) {
            *to++ = *from--;
        }
        dest += keys - len_buf;

        // Compute the length (in bytes) and return length in bytes
        return dest - (uint8_t*)dest_as_void;
    }

    void decode(uint32_t* to, const void* src, size_t len)
    {
        __m128i byte_stream, byte_stream_2, tmp, tmp2, mask_21, mask_12, mask_10, mask_9, mask_7, mask_6, mask_5,
            mask_4, mask_3, mask_2, mask_1;
        uint8_t* in = (uint8_t*)src;
        uint8_t* keys = ((uint8_t*)src) + len - 1;

        mask_21 = _mm_loadu_si128((__m128i*)constants::static_mask_21);
        mask_12 = _mm_loadu_si128((__m128i*)constants::static_mask_12);
        mask_10 = _mm_loadu_si128((__m128i*)constants::static_mask_10);
        mask_9 = _mm_loadu_si128((__m128i*)constants::static_mask_9);
        mask_7 = _mm_loadu_si128((__m128i*)constants::static_mask_7);
        mask_6 = _mm_loadu_si128((__m128i*)constants::static_mask_6);
        mask_5 = _mm_loadu_si128((__m128i*)constants::static_mask_5);
        mask_4 = _mm_loadu_si128((__m128i*)constants::static_mask_4);
        mask_3 = _mm_loadu_si128((__m128i*)constants::static_mask_3);
        mask_2 = _mm_loadu_si128((__m128i*)constants::static_mask_2);
        mask_1 = _mm_loadu_si128((__m128i*)constants::static_mask_1);

        while (in <= keys) {
            switch (*keys--) {
            case 0x00:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x01:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x02:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x03:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x04:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x05:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x06:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x07:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x08:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x09:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x0a:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x0b:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x0c:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x0d:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x0e:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
            case 0x0f:
                tmp = _mm_loadu_si128((__m128i*)constants::static_mask_1);
                _mm_storeu_si128((__m128i*)to, tmp);
                _mm_storeu_si128((__m128i*)to + 1, tmp);
                _mm_storeu_si128((__m128i*)to + 2, tmp);
                _mm_storeu_si128((__m128i*)to + 3, tmp);
                _mm_storeu_si128((__m128i*)to + 4, tmp);
                _mm_storeu_si128((__m128i*)to + 5, tmp);
                _mm_storeu_si128((__m128i*)to + 6, tmp);
                _mm_storeu_si128((__m128i*)to + 7, tmp);
                _mm_storeu_si128((__m128i*)to + 8, tmp);
                _mm_storeu_si128((__m128i*)to + 9, tmp);
                _mm_storeu_si128((__m128i*)to + 10, tmp);
                _mm_storeu_si128((__m128i*)to + 11, tmp);
                _mm_storeu_si128((__m128i*)to + 12, tmp);
                _mm_storeu_si128((__m128i*)to + 13, tmp);
                _mm_storeu_si128((__m128i*)to + 14, tmp);
                _mm_storeu_si128((__m128i*)to + 15, tmp);
                _mm_storeu_si128((__m128i*)to + 16, tmp);
                _mm_storeu_si128((__m128i*)to + 17, tmp);
                _mm_storeu_si128((__m128i*)to + 18, tmp);
                _mm_storeu_si128((__m128i*)to + 19, tmp);
                _mm_storeu_si128((__m128i*)to + 20, tmp);
                _mm_storeu_si128((__m128i*)to + 21, tmp);
                _mm_storeu_si128((__m128i*)to + 22, tmp);
                _mm_storeu_si128((__m128i*)to + 23, tmp);
                _mm_storeu_si128((__m128i*)to + 24, tmp);
                _mm_storeu_si128((__m128i*)to + 25, tmp);
                _mm_storeu_si128((__m128i*)to + 26, tmp);
                _mm_storeu_si128((__m128i*)to + 27, tmp);
                _mm_storeu_si128((__m128i*)to + 28, tmp);
                _mm_storeu_si128((__m128i*)to + 29, tmp);
                _mm_storeu_si128((__m128i*)to + 30, tmp);
                _mm_storeu_si128((__m128i*)to + 31, tmp);
                _mm_storeu_si128((__m128i*)to + 32, tmp);
                _mm_storeu_si128((__m128i*)to + 33, tmp);
                _mm_storeu_si128((__m128i*)to + 34, tmp);
                _mm_storeu_si128((__m128i*)to + 35, tmp);
                _mm_storeu_si128((__m128i*)to + 36, tmp);
                _mm_storeu_si128((__m128i*)to + 37, tmp);
                _mm_storeu_si128((__m128i*)to + 38, tmp);
                _mm_storeu_si128((__m128i*)to + 39, tmp);
                _mm_storeu_si128((__m128i*)to + 40, tmp);
                _mm_storeu_si128((__m128i*)to + 41, tmp);
                _mm_storeu_si128((__m128i*)to + 42, tmp);
                _mm_storeu_si128((__m128i*)to + 43, tmp);
                _mm_storeu_si128((__m128i*)to + 44, tmp);
                _mm_storeu_si128((__m128i*)to + 45, tmp);
                _mm_storeu_si128((__m128i*)to + 46, tmp);
                _mm_storeu_si128((__m128i*)to + 47, tmp);
                _mm_storeu_si128((__m128i*)to + 48, tmp);
                _mm_storeu_si128((__m128i*)to + 49, tmp);
                _mm_storeu_si128((__m128i*)to + 50, tmp);
                _mm_storeu_si128((__m128i*)to + 51, tmp);
                _mm_storeu_si128((__m128i*)to + 52, tmp);
                _mm_storeu_si128((__m128i*)to + 53, tmp);
                _mm_storeu_si128((__m128i*)to + 54, tmp);
                _mm_storeu_si128((__m128i*)to + 55, tmp);
                _mm_storeu_si128((__m128i*)to + 56, tmp);
                _mm_storeu_si128((__m128i*)to + 57, tmp);
                _mm_storeu_si128((__m128i*)to + 58, tmp);
                _mm_storeu_si128((__m128i*)to + 59, tmp);
                _mm_storeu_si128((__m128i*)to + 60, tmp);
                _mm_storeu_si128((__m128i*)to + 61, tmp);
                _mm_storeu_si128((__m128i*)to + 62, tmp);
                _mm_storeu_si128((__m128i*)to + 63, tmp);
                to += 256;
                break;
            case 0x10:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x11:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x12:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x13:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x14:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x15:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x16:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x17:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x18:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x19:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x1a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x1b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x1c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x1d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x1e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
            case 0x1f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 16, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 17, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 18, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 19, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 20, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 21, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 22, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 23, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 24, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 25, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 26, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 27, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 28, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 29, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 30, _mm_and_si128(byte_stream, mask_1));
                byte_stream = _mm_srli_epi64(byte_stream, 1);
                _mm_storeu_si128((__m128i*)to + 31, _mm_and_si128(byte_stream, mask_1));
                in += 16;
                to += 128;
                break;
            case 0x20:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x21:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x22:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x23:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x24:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x25:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x26:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x27:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x28:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x29:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x2a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x2b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x2c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x2d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x2e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
            case 0x2f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 10, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 11, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 12, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 13, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 14, _mm_and_si128(byte_stream, mask_2));
                byte_stream = _mm_srli_epi64(byte_stream, 2);
                _mm_storeu_si128((__m128i*)to + 15, _mm_and_si128(byte_stream, mask_2));
                in += 16;
                to += 64;
                break;
            case 0x30:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x31:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x32:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x33:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x34:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x35:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x36:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x37:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x38:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x39:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x3a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x3b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x3c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x3d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x3e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
            case 0x3f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_3));
                byte_stream = _mm_srli_epi64(byte_stream, 3);
                _mm_storeu_si128((__m128i*)to + 9, _mm_and_si128(byte_stream, mask_3));
                in += 16;
                to += 40;
                break;
            case 0x40:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x41:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x42:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x43:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x44:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x45:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x46:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x47:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x48:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x49:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x4a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x4b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x4c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x4d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x4e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
            case 0x4f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_4));
                byte_stream = _mm_srli_epi64(byte_stream, 4);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_4));
                in += 16;
                to += 32;
                break;
            case 0x50:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x51:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x52:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x53:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x54:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x55:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x56:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x57:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x58:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x59:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x5a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x5b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x5c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x5d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x5e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
            case 0x5f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_5));
                byte_stream = _mm_srli_epi64(byte_stream, 5);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_5));
                in += 16;
                to += 24;
                break;
            case 0x60:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x61:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x62:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x63:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x64:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x65:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x66:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x67:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x68:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x69:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x6a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x6b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x6c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x6d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x6e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
            case 0x6f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_6));
                byte_stream = _mm_srli_epi64(byte_stream, 6);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_6));
                in += 16;
                to += 20;
                break;
            case 0x70:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x71:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x72:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x73:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x74:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x75:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x76:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x77:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x78:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x79:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x7a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x7b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x7c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x7d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x7e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
            case 0x7f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_7));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 4,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 4), _mm_srli_epi32(byte_stream, 7)), mask_7));
                byte_stream = _mm_srli_epi32(byte_stream_2, 3);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 7, _mm_and_si128(byte_stream, mask_7));
                byte_stream = _mm_srli_epi32(byte_stream, 7);
                _mm_storeu_si128((__m128i*)to + 8, _mm_and_si128(byte_stream, mask_7));
                in += 32;
                to += 36;
                break;
            case 0x80:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x81:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x82:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x83:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x84:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x85:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x86:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x87:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x88:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x89:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x8a:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x8b:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x8c:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x8d:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x8e:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
            case 0x8f:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 1, _mm_cvtepu8_epi32(tmp2));
                tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
                _mm_storeu_si128((__m128i*)to + 2, _mm_cvtepu8_epi32(tmp));
                tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp), 0x01));
                _mm_storeu_si128((__m128i*)to + 3, _mm_cvtepu8_epi32(tmp2));
                in += 16;
                to += 16;
                break;
            case 0x90:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x91:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x92:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x93:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x94:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x95:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x96:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x97:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x98:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x99:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x9a:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x9b:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x9c:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x9d:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x9e:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
            case 0x9f:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_9));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 3,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 5), _mm_srli_epi32(byte_stream, 9)), mask_9));
                byte_stream = _mm_srli_epi32(byte_stream_2, 4);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 5, _mm_and_si128(byte_stream, mask_9));
                byte_stream = _mm_srli_epi32(byte_stream, 9);
                _mm_storeu_si128((__m128i*)to + 6, _mm_and_si128(byte_stream, mask_9));
                in += 32;
                to += 28;
                break;
            case 0xa0:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa1:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa2:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa3:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa4:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa5:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa6:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa7:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa8:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xa9:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xaa:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xab:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xac:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xad:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xae:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
            case 0xaf:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_10));
                byte_stream = _mm_srli_epi64(byte_stream, 10);
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(byte_stream, mask_10));
                in += 16;
                to += 12;
                break;
            case 0xb0:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb1:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb2:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb3:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb4:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb5:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb6:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb7:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb8:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xb9:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xba:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xbb:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xbc:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xbd:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xbe:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
            case 0xbf:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 1, _mm_and_si128(byte_stream, mask_12));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 2,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 8), _mm_srli_epi32(byte_stream, 12)), mask_12));
                byte_stream = _mm_srli_epi32(byte_stream_2, 8);
                _mm_storeu_si128((__m128i*)to + 3, _mm_and_si128(byte_stream, mask_12));
                byte_stream = _mm_srli_epi32(byte_stream, 12);
                _mm_storeu_si128((__m128i*)to + 4, _mm_and_si128(byte_stream, mask_12));
                in += 32;
                to += 20;
                break;
            case 0xc0:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc1:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc2:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc3:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc4:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc5:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc6:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc7:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc8:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xc9:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xca:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xcb:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xcc:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xcd:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xce:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
            case 0xcf:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_cvtepu16_epi32(tmp));
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_cvtepu16_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)))));
                in += 16;
                to += 8;
                break;
            case 0xd0:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd1:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd2:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd3:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd4:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd5:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd6:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd7:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd8:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xd9:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xda:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xdb:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xdc:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xdd:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xde:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
            case 0xdf:
                byte_stream = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, _mm_and_si128(byte_stream, mask_21));
                byte_stream_2 = _mm_loadu_si128((__m128i*)in + 1);
                _mm_storeu_si128((__m128i*)to + 1,
                    _mm_and_si128(
                        _mm_or_si128(_mm_slli_epi32(byte_stream_2, 11), _mm_srli_epi32(byte_stream, 21)), mask_21));
                _mm_storeu_si128((__m128i*)to + 2, _mm_and_si128(_mm_srli_epi32(byte_stream_2, 11), mask_21));
                in += 32;
                to += 12;
                break;
            case 0xe0:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe1:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe2:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe3:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe4:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe5:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe6:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe7:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe8:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xe9:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xea:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xeb:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xec:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xed:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xee:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
            case 0xef:
                tmp = _mm_loadu_si128((__m128i*)in);
                _mm_storeu_si128((__m128i*)to, tmp);
                in += 16;
                to += 4;
                break;
            case 0xf0:
                in++;
            case 0xf1:
                in++;
            case 0xf2:
                in++;
            case 0xf3:
                in++;
            case 0xf4:
                in++;
            case 0xf5:
                in++;
            case 0xf6:
                in++;
            case 0xf7:
                in++;
            case 0xf8:
                in++;
            case 0xf9:
                in++;
            case 0xfa:
                in++;
            case 0xfb:
                in++;
            case 0xfc:
                in++;
            case 0xfd:
                in++;
            case 0xfe:
                in++;
            case 0xff:
                in++;
                break;
            }
        }
    }

    void unittest() {}

}; // end codec struct

} // end qmx namespace

#pragma GCC diagnostic pop