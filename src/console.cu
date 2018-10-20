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

__GURU__
void guru_write(mrbc_vtype tt, mrbc_vtype fmt, size_t sz, uint8_t *buf)
{
	guru_print_node *n = (guru_print_node *)guru_output_ptr;

	MEMCPY((uint8_t *)n->data, buf, sz);

	n->tt   = tt;
	n->fmt  = fmt;
	n->size = (sz + 7) & ~0x7;		// 16

	guru_output_ptr = (uint8_t *)n->data + n->size;		// advance pointer to next print block

	*guru_output_ptr = (uint8_t)MRBC_TT_EMPTY;
}

//================================================================
/*! output a character

  @param  c	character
*/
__GURU__
void console_char(char c)
{
	char buf[2] = { c, '\0' };
	guru_write(MRBC_TT_STRING, MRBC_TT_EMPTY, 2, (uint8_t *)buf);
}

__GURU__
void console_int(mrbc_int i)
{
	guru_write(MRBC_TT_FIXNUM, MRBC_TT_EMPTY, sizeof(mrbc_int), (uint8_t *)&i);
}

__GURU__
void console_hex(mrbc_int i)
{
	guru_write(MRBC_TT_FIXNUM, MRBC_TT_FIXNUM, sizeof(mrbc_int), (uint8_t *)&i);
}

#if MRBC_USE_FLOAT
__GURU__
void console_float(mrbc_float f);
{
	guru_write(MRBC_TT_FLOAT, MRBC_TT_EMPTY, sizeof(mrbc_float), (uint8_t *)&f);
}
#endif

//================================================================
/*! output string

  @param str	str
*/
__GURU__
void console_str(const char *str)
{
	guru_write(MRBC_TT_STRING, MRBC_TT_EMPTY, guru_strlen(str), (uint8_t *)str);
}

__GURU__
void console_strf(const char *str, const char *fmt)
{
	guru_write(MRBC_TT_STRING, MRBC_TT_STRING, guru_strlen(fmt), (uint8_t *)fmt);
	guru_write(MRBC_TT_STRING, MRBC_TT_EMPTY,  guru_strlen(str), (uint8_t *)str);
}

__global__
void guru_console_init(uint8_t *buf, size_t sz)
{
	if (threadIdx.x!=0 || blockIdx.x !=0) return;

	guru_output = guru_output_ptr = buf;
	guru_output_size = sz;
}

#define NEXTNODE(n)	((guru_print_node *)(node->data+node->size))

__host__
void guru_print(uint8_t *output_buf)
{
	guru_print_node *node = (guru_print_node *)output_buf;
	uint8_t *fmt[80], *buf[80];		// check buffer overflow

	while (node->tt != MRBC_TT_EMPTY) {		// 0
		switch (node->tt) {
		case MRBC_TT_FIXNUM:
			printf("%d", *((mrbc_int *)node->data));
			break;
		case MRBC_TT_FLOAT:
			printf("%f", *((mrbc_float *)node->data));
			break;
		case MRBC_TT_STRING:
			if (node->fmt==MRBC_TT_STRING) {
				memcpy(fmt, (uint8_t *)node->data, node->size);
				node = NEXTNODE(node);
			}
			else {
				memcpy(fmt, "%s", sizeof("%s"));
			}
			memcpy(buf, (uint8_t *)node->data, node->size);
			buf[node->size] = '\0';
			printf((const char *)fmt, buf);
			break;
		default: printf("not supported: %d", node->tt); break;
		}
		node = NEXTNODE(node);
	}
}

