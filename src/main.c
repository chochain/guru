#include <stdio.h>
#include "guru.h"

extern int do_cuda(void);

int _memcmp(const uint8_t *d, const uint8_t *s, size_t sz)
{
    int i;

    for (i=0; i<sz && *d++==*s++; i++);

    return i<sz;
}

int load_header(const uint8_t **pos)
{
    const uint8_t *p = *pos;

    if (_memcmp(p, "RITE0004", 8) != 0) {
        return -1;
    }

    /* Ignore CRC */
    /* Ignore size */

    if (_memcmp(p + 14, "MATZ", 4) != 0) {
        return -1;
    }
    if (_memcmp(p + 18, "0000", 4) != 0) {
        return -1;
    }
    *pos += 22;
    return 0;
}

uint32_t bin_to_uint32(const void *s)
{
    uint32_t x = *((uint32_t *)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

int load_irep(const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;			// 4 = skip "RITE"
    int   section_size = bin_to_uint32(p);

    p += sizeof(uint32_t);
    if (_memcmp(p, "0000", 4) != 0) {					// rite version
        return -1;
    }
    p += 4;

    *pos += section_size;
    return 0;
}

void upload_bytecode(const uint8_t *ptr)
{
    int ret;

    ret = load_header(&ptr);
    ret = load_irep(&ptr);

    return;
}

int main(int argc, char **argv)
{
    //do_cuda();

	guru_ses ses;
	int rst = init_session(&ses, argv[1]);

	upload_bytecode((uint8_t *)ses.req);

    return 0;
}
