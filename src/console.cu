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

__GURU__ uint8_t *guru_output;
__GURU__ uint8_t *guru_output_ptr;	// global output buffer for now, per session later

__GURU__
void guru_write(mrbc_vtype tt, size_t sz, const char *fmt, uint8_t *buf)
{
	guru_print_node *n = (guru_print_node *)guru_output_ptr;

	MEMCPY((uint8_t *)n->data, buf, sz);

	n->tt   = tt;
	n->size = (sz + 7) & ~0x7;		// 16
	n->fmt  = fmt;

	guru_output_ptr += sizeof(n->tt) + sizeof(n->size) + sizeof(n->fmt) + n->size;

	*((mrbc_int *)guru_output_ptr) = (mrbc_int)MRBC_TT_EMPTY;		// 0
}

//================================================================
/*! output a character

  @param  c	character
*/
__GURU__
void console_char(char c)
{
	char buf[2] = { c, '\0' };
	guru_write(MRBC_TT_STRING, 2, "%s", (uint8_t *)buf);
}

__GURU__
void console_int(mrbc_int i)
{
	guru_write(MRBC_TT_FIXNUM, sizeof(mrbc_int), "%d", (uint8_t *)&i);
}

__GURU__
void console_hex(mrbc_int i)
{
	guru_write(MRBC_TT_FIXNUM, sizeof(mrbc_int), "0x%x", (uint8_t *)&i);
}

#if MRBC_USE_FLOAT
__GURU__
void console_float(mrbc_float f);
{
	guru_write(MRBC_TT_FIXNUM, sizeof(mrbc_float), "%f", (uint8_t *)&f);
}
#endif

//================================================================
/*! output string

  @param str	str
*/
__GURU__
void console_str(const char *str)
{
	guru_write(MRBC_TT_STRING, guru_strlen(str), "%s", (uint8_t *)str);
}

__GURU__
void console_strf(const char *str, const char *fmt)
{
	guru_write(MRBC_TT_STRING, guru_strlen(str), fmt, (uint8_t *)str);
}

__global__
void guru_init_console_buf(uint8_t *buf, size_t sz)
{
	if (threadIdx.x!=0 || blockIdx.x !=0) return;

	guru_output = guru_output_ptr = buf;
}

__host__
void guru_print(uint8_t *output_buf)
{
	guru_print_node *node = (guru_print_node *)output_buf;
	uint8_t *buf[80];		// check buffer overflow

	while (node->tt != MRBC_TT_EMPTY) {		// 0
		switch (node->tt) {
		case MRBC_TT_FIXNUM:
			printf(node->fmt, *((mrbc_int *)node->data));
			break;
		case MRBC_TT_FLOAT:
			printf(node->fmt, *((mrbc_float *)node->data));
			break;
		case MRBC_TT_STRING:
			memcpy(buf, (uint8_t *)node->data, node->size);
			buf[node->size] = '\0';
			printf(node->fmt, buf);
			break;
		default: printf("not supported: %d", node->tt); break;
		}
		node = (guru_print_node *)(node->data+node->size);
	}
}

