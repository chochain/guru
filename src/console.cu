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

__GURU__ U32 _output_size;
__GURU__ U8  *_output_ptr;		// global output buffer for now, per session later
__GURU__ U8  *_output_buf;

__GURU__ volatile U32 _mutex_con;

__GURU__ void
guru_write(mrbc_vtype tt, mrbc_vtype fmt, U32 sz, U8 *buf)
{
	if (threadIdx.x!=0) return;		// only thread 0 within a block can write

	MUTEX_LOCK(_mutex_con);

	guru_print_node *n = (guru_print_node *)_output_ptr;
	MEMCPY((U8 *)n->data, buf, sz);

	n->id   = blockIdx.x;			// VM.id
	n->tt   = tt;
	n->fmt  = fmt;
	n->size = sz + (-sz & 0x3);		// 32-bit alignment

	_output_ptr  = (U8 *)n->data + n->size;		// advance pointer to next print block
	*_output_ptr = (U8)GURU_TT_EMPTY;

	MUTEX_FREE(_mutex_con);
}

//================================================================
/*! output a character

  @param  c	character
*/
__GURU__ void
console_char(U8 c)
{
	U8 buf[2] = { c, '\0' };
	guru_write(GURU_TT_STRING, GURU_TT_EMPTY, 2, buf);
}

__GURU__ void
console_int(mrbc_int i)
{
	guru_write(GURU_TT_FIXNUM, GURU_TT_FIXNUM, sizeof(mrbc_int), (U8 *)&i);
}

__GURU__ void
console_hex(mrbc_int i)
{
	guru_write(GURU_TT_FIXNUM, GURU_TT_EMPTY, sizeof(mrbc_int), (U8 *)&i);
}

#if GURU_USE_FLOAT
__GURU__ void
console_float(mrbc_float f)
{
	guru_write(GURU_TT_FLOAT, GURU_TT_EMPTY, sizeof(mrbc_float), (U8 *)&f);
}
#endif

//================================================================
/*! output string

  @param str	str
*/
__GURU__ void
console_str(const U8 *str)
{
	guru_write(GURU_TT_STRING, GURU_TT_EMPTY, guru_strlen(str)+1, (U8 *)str);
}

__GURU__ void
console_na(const U8 *msg)
{
    console_str("method not supported: ");
	console_str(msg);
	console_str("\n");
}

__GURU__ U8*
_console_va_arg(U8 *p)
{
    U8 ch;
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
    S32 n;
    while ((n = *p - '0'), (0 <= n & n <= 9)) p++;
    if (*p == '.') {
        p++;
        while ((n = *p - '0'), (0 <= n && n <= 9)) p++;
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
	console_int(sizeof(RVar));
    console_str("\nproc=");
    console_int(sizeof(RProc));
    console_str("\nstring=");
    console_int(sizeof(RString));
    console_str("\n");
}

__GPU__ void
guru_console_init(U8 *buf, U32 sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	guru_print_node *node = (guru_print_node *)buf;
	node->tt = GURU_TT_EMPTY;

	_output_buf = _output_ptr = buf;

	if (sz) _output_size = sz;					// set to new size
}

#define NEXTNODE(n)	((guru_print_node *)(node->data + node->size))

__HOST__ guru_print_node*
_guru_host_print(guru_print_node *node)
{
	U8  *fmt[80], *buf[80];					// check buffer overflow
	U32	argc;
	printf("<%d>", node->id);
	switch (node->tt) {
	case GURU_TT_FIXNUM:
		printf((node->fmt==GURU_TT_FIXNUM ? "%d" : "%04x"), *((mrbc_int *)node->data));
		break;
	case GURU_TT_FLOAT:
		printf("%g", *((mrbc_float *)node->data));
		break;
	case GURU_TT_STRING:
		memcpy(buf, (U8 *)node->data, node->size);
		printf("%s", (char *)buf);
		break;
	case GURU_TT_RANGE:							// TODO: va_list needed here
		argc = (int)node->fmt;
		memcpy(fmt, (U8 *)node->data, node->size);
		printf("%s", (char *)fmt);
		for (int i=0; i<argc; i++) {
			node = NEXTNODE(node);				// point to next parameter
			node = _guru_host_print(node);		// recursive call
		}
		break;
	default: printf("print node type not supported: %d", node->tt); break;
	}
	printf("</%d>\n", node->id);
	return node;
}

__HOST__ void
guru_console_flush(U8 *output_buf)
{
	guru_print_node *node = (guru_print_node *)output_buf;
	while (node->tt != GURU_TT_EMPTY) {			// 0
		node = _guru_host_print(node);
		node = NEXTNODE(node);
	}
	guru_console_init<<<1,1>>>(output_buf, 0);
	cudaDeviceSynchronize();
}

