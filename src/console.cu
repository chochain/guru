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
__GURU__ U8P _output_ptr;		// global output buffer for now, per session later
__GURU__ U8P _output_buf;

__GURU__ volatile U32 _mutex_con;

__GURU__ void
_write(GT gt, GT fmt, U32 sz, U8P buf)
{
	if (threadIdx.x!=0) return;		// only thread 0 within a block can write

	MUTEX_LOCK(_mutex_con);

	guru_print_node *n = (guru_print_node *)_output_ptr;
	MEMCPY((U8P)n->data, buf, sz);

	n->id   = blockIdx.x;			// VM.id
	n->gt   = gt;
	n->fmt  = fmt;
	n->size = sz + (-sz & 0x3);		// 32-bit alignment

	_output_ptr  = U8PADD(n->data, n->size);		// advance pointer to next print block
	*_output_ptr = (U8)GT_EMPTY;

	MUTEX_FREE(_mutex_con);
}

__GURU__ U8P
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
	console_int(sizeof(GV));
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
guru_console_init(U8P buf, U32 sz)
{
#if GURU_USE_CONSOLE
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	guru_print_node *node = (guru_print_node *)buf;
	node->gt = GT_EMPTY;

	_output_buf = _output_ptr = buf;

	if (sz) _output_size = sz;					// set to new size
#endif
}

#define NEXTNODE(n)	((guru_print_node *)(node->data + node->size))

//================================================================
/*! output a character

  @param  c	character
*/
__GURU__ void
console_char(U8 c)
{
	U8 buf[2] = { c, '\0' };
	_write(GT_STR, GT_EMPTY, 2, (U8P)buf);
}

__GURU__ void
console_int(GI i)
{
	_write(GT_INT, GT_INT, sizeof(GI), (U8P)&i);
}

__GURU__ void
console_hex(GI i)
{
	_write(GT_INT, GT_EMPTY, sizeof(GI), (U8P)&i);
}

__GURU__ void
console_ptr(void *ptr)
{
    console_hex((U32A)ptr>>16);
    console_hex((U32A)ptr&0xffff);
}

#if GURU_USE_FLOAT
__GURU__ void
console_float(GF f)
{
	_write(GT_FLOAT, GT_EMPTY, sizeof(GF), (U8P)&f);
}
#endif

//================================================================
/*! output string

  @param str	str
*/
__GURU__ void
console_str(const U8 *str)
{
	_write(GT_STR, GT_EMPTY, guru_strlen((U8P)str)+1, (U8P)str);
}

__HOST__ guru_print_node*
_guru_host_print(guru_print_node *node, U32 trace)
{
	U8P fmt[80];
	U8P buf[80];								// check buffer overflow
	U32	argc;

	if (trace) printf("<%d>", node->id);
	switch (node->gt) {
	case GT_INT:
		printf((node->fmt==GT_INT ? "%d" : "%04x"), *((GI *)node->data));
		break;
	case GT_FLOAT:
		printf("%g", *((GF *)node->data));
		break;
	case GT_STR:
		memcpy(buf, (U8P)node->data, node->size);
		printf("%s", (char *)buf);
		break;
	case GT_RANGE:								// TODO: va_list needed here
		argc = (int)node->fmt;
		memcpy(fmt, (U8P)node->data, node->size);
		printf("%s", (char *)fmt);
		for (U32 i=0; i<argc; i++) {
			node = NEXTNODE(node);					// point to next parameter
			node = _guru_host_print(node, trace);	// recursive call
		}
		break;
	default: printf("print node type not supported: %d", node->gt); break;
	}

	if (trace) printf("</%d>\n", node->id);

	return node;
}

__HOST__ void
guru_console_flush(U8P output_buf, U32 trace)
{
#if GURU_USE_CONSOLE
	guru_print_node *node = (guru_print_node *)output_buf;
	while (node->gt != GT_EMPTY) {			// 0
		node = _guru_host_print(node, trace);
		node = NEXTNODE(node);
	}
	guru_console_init<<<1,1>>>(output_buf, 0);
	cudaDeviceSynchronize();
#endif
}
