#pragma once

#include <vector>

namespace ds2i {

class Varint {
   public:

    template <uint32_t i>
    static uint8_t extract7bits(const uint32_t val) {
        return static_cast<uint8_t>((val >> (7 * i)) & ((1U << 7) - 1));
    }

    template <uint32_t i>
    static uint8_t extract7bitsmaskless(const uint32_t val) {
        return static_cast<uint8_t>((val >> (7 * i)));
    }

    static void encode(const uint32_t *in, const size_t length, uint8_t *out, size_t &nvalue) {
        uint8_t *bout = out;
        for (size_t k = 0; k < length; ++k) {
            const uint32_t val(in[k]);
            /**
             * Code below could be shorter. Whether it could be faster
             * depends on your compiler and machine.
             */
            if (val < (1U << 7)) {
                *bout = static_cast<uint8_t>(val | (1U << 7));
                ++bout;
            } else if (val < (1U << 14)) {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bitsmaskless<1>(val) | (1U << 7);
                ++bout;
            } else if (val < (1U << 21)) {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bits<1>(val);
                ++bout;
                *bout = extract7bitsmaskless<2>(val) | (1U << 7);
                ++bout;
            } else if (val < (1U << 28)) {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bits<1>(val);
                ++bout;
                *bout = extract7bits<2>(val);
                ++bout;
                *bout = extract7bitsmaskless<3>(val) | (1U << 7);
                ++bout;
            } else {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bits<1>(val);
                ++bout;
                *bout = extract7bits<2>(val);
                ++bout;
                *bout = extract7bits<3>(val);
                ++bout;
                *bout = extract7bitsmaskless<4>(val) | (1U << 7);
                ++bout;
            }
        }
        nvalue = bout - out;
    }

    static void encode_single(uint32_t val, std::vector<uint8_t> &out) {
        uint8_t buf[5];
        size_t  nvalue;
        encode(&val, 1, buf, nvalue);
        out.insert(out.end(), buf, buf + nvalue);
    }

    static size_t decode(const uint8_t *in, uint32_t *out, size_t len) {
        const uint8_t *inbyte = in;
        size_t n = 0;
        while (inbyte < in + len) {
            unsigned int shift = 0;
            for (uint32_t v = 0;; shift += 7) {
                uint8_t c = *inbyte++;
                v += ((c & 127) << shift);
                if ((c & 128)) {
                    *out++ = v;
                    n += 1;
                    break;
                }
            }
        }
        return n;
    }
};
} // namespace ds2i
