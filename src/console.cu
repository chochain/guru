/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdio.h>
#include "value.h"
#include "console.h"

__GURU__ size_t  guru_output_size;
__GURU__ uint8_t *guru_output;
__GURU__ uint8_t *guru_output_ptr;	// global output buffer for now, per session later

__GURU__ void
guru_write(mrbc_vtype tt, mrbc_vtype fmt, size_t sz, uint8_t *buf)
{
	guru_print_node *n = (guru_print_node *)guru_output_ptr;

	MEMCPY((uint8_t *)n->data, buf, sz);

	n->tt   = tt;
	n->fmt  = fmt;
	n->size = sz + (-sz & 0x3);		// 32-bit alignment

	guru_output_ptr = (uint8_t *)n->data + n->size;		// advance pointer to next print block

	*guru_output_ptr = (uint8_t)MRBC_TT_EMPTY;
}

//================================================================
/*! output a character

  @param  c	character
*/
__GURU__ void
console_char(char c)
{
	char buf[2] = { c, '\0' };
	guru_write(MRBC_TT_STRING, MRBC_TT_EMPTY, 2, (uint8_t *)buf);
}

__GURU__ void
console_int(mrbc_int i)
{
	guru_write(MRBC_TT_FIXNUM, MRBC_TT_FIXNUM, sizeof(mrbc_int), (uint8_t *)&i);
}

__GURU__ void
console_hex(mrbc_int i)
{
	guru_write(MRBC_TT_FIXNUM, MRBC_TT_EMPTY, sizeof(mrbc_int), (uint8_t *)&i);
}

#if MRBC_USE_FLOAT
__GURU__ void
console_float(mrbc_float f)
{
	guru_write(MRBC_TT_FLOAT, MRBC_TT_EMPTY, sizeof(mrbc_float), (uint8_t *)&f);
}
#endif

//================================================================
/*! output string

  @param str	str
*/
__GURU__ void
console_str(const char *str)
{
	guru_write(MRBC_TT_STRING, MRBC_TT_EMPTY, guru_strlen(str)+1, (uint8_t *)str);
}

__GURU__ void
console_na(const char *msg)
{
	console_str(msg);
    console_str(" not supported!\n");
}

__GURU__ char*
_console_va_arg(char *p)
{
    int ch;
    while ((ch = *p) != '\0') {
        p++;
        if (ch == '%') {
            if (*p == '%') p++;	// is "%%"
            else 	       goto PARSE_FLAG;
        }
    }
    if (ch == '\0') return NULL;

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.precision] type
    //   e.g. "%05d"
    while ((ch = *p)) {
        switch(ch) {
        case '+': case ' ': case '-': case '0': break;
        default : goto PARSE_WIDTH;
        }
        p++;
    }

PARSE_WIDTH:
    while ((ch = *p - '0'), (0 <= ch && ch <= 9)) p++;
    if (*p == '.') {
        p++;
        while ((ch = *p - '0'), (0 <= ch && ch <= 9)) p++;
    }
    if (*p) ch = *p++;

    return p;
}

__GURU__ void
_dump_obj_size(void)
{
	console_str("\nvalue=");
	console_int(sizeof(mrbc_value));
	console_str("\nrclass=");
	console_int(sizeof(RClass));
	console_str("\ninstance=");
	console_int(sizeof(RInstance));
    console_str("\nproc=");
    console_int(sizeof(RProc));
    console_str("\nstring=");
    console_int(sizeof(RString));
    console_str("\n");
}

__global__ void
guru_console_init(uint8_t *buf, size_t sz)
{
	if (threadIdx.x!=0 || blockIdx.x !=0) return;

	guru_print_node *node = (guru_print_node *)buf;
	node->tt = MRBC_TT_EMPTY;

	guru_output = guru_output_ptr = buf;

	if (sz) guru_output_size = sz;					// set to new size
}

#define NEXTNODE(n)	((guru_print_node *)(node->data + node->size))

__host__ guru_print_node*
_guru_print_core(guru_print_node *node)
{
	uint8_t *fmt[80], *buf[80];		// check buffer overflow
	int 	argc;
	switch (node->tt) {
	case MRBC_TT_FIXNUM:
		printf((node->fmt==MRBC_TT_FIXNUM ? "%d" : "%04x"), *((mrbc_int *)node->data));
		break;
	case MRBC_TT_FLOAT:
		printf("%f", *((mrbc_float *)node->data));
		break;
	case MRBC_TT_STRING:
		memcpy(buf, (uint8_t *)node->data, node->size);
		printf("%s", (char *)buf);
		break;
	case MRBC_TT_RANGE:							// TODO: va_list needed here
		argc = (int)node->fmt;
		memcpy(fmt, (uint8_t *)node->data, node->size);
		printf("%s", (char *)fmt);
		for (int i=0; i<argc; i++) {
			node = NEXTNODE(node);				// point to next parameter
			node = _guru_print_core(node);		// recursive call
		}
		break;
	default: printf("not supported: %d", node->tt); break;
	}
	return node;
}

__host__ void
guru_console_flush(uint8_t *output_buf)
{
	guru_print_node *node = (guru_print_node *)output_buf;

	while (node->tt != MRBC_TT_EMPTY) {			// 0
		node = _guru_print_core(node);
		node = NEXTNODE(node);
	}
	guru_console_init<<<1,1>>>(output_buf, 0);
	cudaDeviceSynchronize();
}

