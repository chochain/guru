/*! @file
  @brief
  GURU console output module. (not yet input)

  guru_config.h#GURU_USE_CONSOLE can switch between CUDA or internal implementation
  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdio.h>
#include <stdarg.h>

#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "symbol.h"
#include "console.h"
#include "sprintf.h"

#define PRINT_BUFSIZE	128		// local memory

//================================================================
/*! initialize data container.

  @param  pf	pointer to guru_print
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
  @param  fstr	format string.
*/
__GURU__ guru_print *
_init(guru_print *pf, U8 *buf, U32 sz, const U8 *fstr)
{
    pf->fmt = (guru_print_fmt){0};
    pf->fstr= fstr;
    pf->end = buf + sz;			// buf array boundary
    pf->buf = pf->p = buf;		// point at the start of buf

    return pf;
}

//================================================================
/*! return string length in buffer

  @param  pf	pointer to guru_print
  @return	length
*/
__GURU__ __INLINE__ S32
_size(guru_print *pf)
{
    return pf->end - pf->p;
}

//================================================================
/*! terminate ('\0') output buffer.

  @param  pf	pointer to guru_print
*/
__GURU__ __INLINE__ void
_done(guru_print *pf)
{
    *pf->p = '\0';
}

//================================================================
/*! sprintf subcontract function

  @param  pf	pointer to guru_print
  @retval 0	(format string) done.
  @retval 1	found a format identifier.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ int
_next(guru_print *pf)
{
    U8  ch = '\0';
    pf->fmt = (guru_print_fmt){0};

    while (_size(pf) && (ch = *pf->fstr) != '\0') {
        pf->fstr++;
        if (ch == '%') {
            if (*pf->fstr == '%') {	// is "%%"
                pf->fstr++;
            }
            else goto PARSE_FLAG;
        }
        *pf->p++ = ch;
    }
    return -(_size(pf) && ch != '\0');

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.prec] type
    //   e.g. "%05d"
    while ((ch = *(pf->fstr))) {
        switch(ch) {
        case '+': pf->fmt.plus  = 1; break;
        case ' ': pf->fmt.space = 1; break;
        case '-': pf->fmt.minus = 1; break;
        case '0': pf->fmt.zero  = 1; break;
        default : goto PARSE_WIDTH;
        }
        pf->fstr++;
    }

PARSE_WIDTH:
	S32 n;
    while ((n = *pf->fstr - '0'), (0 <= n && n <= 9)) {	// isdigit()
        pf->fmt.width = pf->fmt.width * 10 + n;
        pf->fstr++;
    }
    if (*pf->fstr == '.') {
        pf->fstr++;
        while ((n = *pf->fstr - '0'), (0 <= n && n <= 9)) {
            pf->fmt.prec = pf->fmt.prec * 10 + n;
            pf->fstr++;
        }
    }
    if (*pf->fstr) pf->fmt.type = *pf->fstr++;

    return 1;
}

//================================================================
/*! sprintf subcontract function for U8 '%c'

  @param  pf	pointer to guru_print
  @param  ch	output character (ASCII)
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__char(guru_print *pf, U8 ch)
{
    if (pf->fmt.minus) {
        if (_size(pf)) *pf->p++ = ch;
        else return -1;
    }
    for (U32 i=0; i < pf->fmt.width; i++) {
        if (_size(pf)) *pf->p++ = ' ';
        else return -1;
    }
    if (!pf->fmt.minus) {
        if (_size(pf)) *pf->p++ = ch;
        else return -1;
    }
    return 0;
}

#if GURU_USE_FLOAT
//================================================================
/*! sprintf subcontract function for float(double) '%f'

  @param  pf	pointer to guru_print.
  @param  value	output value.
  @retval 0	done.
  @retval -1	buffer full.
*/
__GURU__ int
__float(guru_print *pf, double value)
{
    U8  fstr[16];
    U8 *p0 = (U8*)pf->fstr;
    U8 *p1 = fstr + sizeof(fstr) - 1;

    *p1 = '\0';
    while ((*--p1 = *--p0) != '%');

    // TODO: 20181025 format print float
    //snprintf(pf->p, (pf->buf_end - pf->p + 1), p1, value);

    while (*pf->p != '\0')
        pf->p++;

    return _size(pf);
}
#endif // GURU_USE_FLOAT

//================================================================
/*! sprintf subcontract function for U8 '%s'

  @param  pf	pointer to guru_print.
  @param  str	output string.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__str(guru_print *pf, U8 *str, U8 pad)
{
	U32 len = STRLENB(str);
    S32 ret = 0;

    if (str == NULL) {
        str = (U8*)"(null)";
        len = 6;
    }
    if (pf->fmt.prec && len > pf->fmt.prec) len = pf->fmt.prec;

    S32 tw = len;
    if (pf->fmt.width > len) tw = pf->fmt.width;

    ASSERT(len <= _size(pf));
    ASSERT(tw  <= _size(pf));

    S32 n_pad = tw - len;

    if (!pf->fmt.minus) {							// right padding
    	MEMSET(pf->p, pad, n_pad);	pf->p += n_pad;
    }
    MEMCPY(pf->p, str, len);	pf->p += len;
    MEMSET(pf->p, pad, n_pad);	pf->p += n_pad;		// left padding

    return ret;
}

//================================================================
/*! sprintf subcontract function for integer '%d' '%x' '%b'

  @param  pf	pointer to guru_print.
  @param  value	output value.
  @param  base	n base.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__int(guru_print *pf, GI value, U32 base)
{
    U32 sign = 0;
    U32 v = value;			// (note) Change this when supporting 64 bit.

    if (pf->fmt.type == 'd' || pf->fmt.type == 'i') {	// signed.
        if (value < 0) {
            sign = '-';
            v = -value;
        } else if (pf->fmt.plus) {
            sign = '+';
        } else if (pf->fmt.space) {
            sign = ' ';
        }
    }
    if (pf->fmt.minus || pf->fmt.width == 0) {
        pf->fmt.zero = 0; 	// disable zero padding if left align or width zero.
    }
    pf->fmt.prec = 0;

    U32 bias_a = (pf->fmt.type == 'X') ? 'A' - 10 : 'a' - 10;

    // create string to local buffer
    U8 buf[64+2];				// int64 + terminate + 1
    U8 *p = buf + sizeof(buf) - 1;
    *p = '\0';
    do {
        U32 i = v % base;
        *--p = (i < 10)? i + '0' : i + bias_a;
        v /= base;
    } while (v != 0);

    // decide pad character and output sign character
    U8 pad;
    if (pf->fmt.zero) {
        pad = '0';
        if (sign) {
            *pf->p++ = sign;
            if (!_size(pf)) return -1;
            pf->fmt.width--;
        }
    } else {
        pad = ' ';
        if (sign) *--p = sign;
    }
    return __str(pf, (U8*)p, pad);
}

//================================================================
/*! output formatted string

  @param  fstr		format string.
*/
__GURU__ void
guru_printf(const U8 *fstr, ...)
{
	U8 buf[PRINT_BUFSIZE];
    va_list ap;
    va_start(ap, fstr);

    U32 ret = 0;
    guru_print pa, *pf = _init(&pa, buf, PRINT_BUFSIZE, fstr);

    while (ret==0 && _next(pf)) {
    	switch(pf->fmt.type) {
        case 'c': ret = __char(pf, va_arg(ap, int));        	 	break;
        case 's': ret = __str(pf, va_arg(ap, U8 *), ' '); 	 		break;
        case 'd':
        case 'i':
        case 'u': ret = __int(pf, va_arg(ap, unsigned int), 10); 	break;
        case 'b':
        case 'B': ret = __int(pf, va_arg(ap, unsigned int), 2);  	break;
        case 'x':
        case 'X': ret = __int(pf, va_arg(ap, unsigned int), 16); 	break;
#if GURU_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G': ret = __float(pf, va_arg(ap, double)); 		 	break;
#endif // GURU_USE_FLOAT
        default:
            console_str("?format: ");
            console_char(pf->fmt.type);
            console_str("\n");
            break;
        }
    }
    va_end(ap);
    _done(pf);

    console_str(pf->buf);
}

__GURU__ void
guru_vprintf(const U8 *fstr, GV v[], U32 vi)		// << from c_string.cu
{
	U8  buf[PRINT_BUFSIZE];
    U32 i   = 0;
    U32 ret = 0;
    guru_print pa, *pf = _init(&pa, buf, PRINT_BUFSIZE, fstr);

    while (ret==0 && _next(pf)) {
        if (i > vi) {
        	console_str("#guru_vprint ArgumentError\n");
        }
        switch(pf->fmt.type) {
        case 'c':
            if (v[i].gt==GT_INT) {
                ret = __char(pf, v[i].i);
            }
            break;
        case 's':
            if (v[i].gt==GT_STR) {
                ret = __str(pf, (U8*)v[i].str->raw, ' ');
            }
            else if (v[i].gt==GT_SYM) {
                ret = __str(pf, id2name(v[i].i), ' ');
            }
            break;
        case 'd':
        case 'i':
        case 'u':
            if (v[i].gt==GT_INT) {
                ret = __int(pf, v[i].i, 10);
#if GURU_USE_FLOAT
            } else if (v[i].gt==GT_FLOAT) {
                ret = __int(pf, (GI)v[i].f, 10);
#endif // GURU_USE_FLOAT
            } else if (v[i].gt==GT_STR) {
                GI ival = ATOI(v[i].str->raw, 10);
                ret = __int(pf, ival, 10);
            }
            break;
        case 'b':
        case 'B':
            if (v[i].gt==GT_INT) {
                ret = __int(pf, v[i].i, 2);
            }
            break;
        case 'x':
        case 'X':
            if (v[i].gt==GT_INT) {
                ret = __int(pf, v[i].i, 16);
            }
            break;
#if GURU_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G':
            if (v[i].gt==GT_FLOAT) {
                ret = __float(pf, v[i].f);
            }
            else if (v[i].gt==GT_INT) {
            	ret = __float(pf, v[i].i);
            }
            break;
#endif // GURU_USE_FLOAT
        default: break;
        }
        i++;
    }
    _done(pf);

    console_str(pf->buf);
}
