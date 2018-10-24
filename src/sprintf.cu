/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "alloc.h"
#include "value.h"
#include "symbol.h"
#include "console.h"
#include "sprintf.h"

#define BUF_STEP_SIZE 80

//================================================================
/*! initialize data container.

  @param  pf	pointer to mrbc_printf
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
  @param  fstr	format string.
*/
__GURU__
void _mrbc_printf_init(mrbc_printf *pf, char *buf, int sz, const char *fstr)
{
    pf->buf = pf->p = buf;    
    pf->end = buf + sz - 1;
    pf->fstr= fstr;
    pf->fmt = (mrbc_print_fmt){0};
}

//================================================================
/*! clear output buffer in container.

  @param  pf	pointer to mrbc_printf
*/
__GURU__ __forceinline__
void _mrbc_printf_clear(mrbc_printf *pf)
{
    pf->p = pf->buf;		// back to head
}

//================================================================
/*! terminate ('\0') output buffer.

  @param  pf	pointer to mrbc_printf
*/
__GURU__ __forceinline__
void _mrbc_printf_end(mrbc_printf *pf)
{
    *pf->p = '\0';
}

//================================================================
/*! return string length in buffer

  @param  pf	pointer to mrbc_printf
  @return	length
*/
__GURU__ __forceinline__
int _mrbc_printf_len(mrbc_printf *pf)
{
    return pf->p - pf->buf;
}

//================================================================
/*! replace output buffer

  @param  pf	pointer to mrbc_printf
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
*/
__GURU__
int _mrbc_printf_resize(mrbc_printf *pf)
{
    int sz  = pf->end - pf->p;
	if (sz > 0) return 1;

    int off = pf->p - pf->buf;					        // current offset
    int inc = BUF_STEP_SIZE;
	while ((sz + inc) < pf->fmt.width) {
		inc += BUF_STEP_SIZE;
	}

	int nsz = (pf->end - pf->buf + 1) + inc;	        // new size
	char *nbuf = (char *)mrbc_realloc(pf->buf, nsz);	// deep copy
	if (!nbuf) return 0;

    pf->buf = nbuf;
    pf->end = nbuf + nsz - 1;                           // last byte for '\0'
    pf->p   = pf->buf + off;				            // reset pointer

    return 1;
}

//================================================================
/*! sprintf subcontract function for char '%c'

  @param  pf	pointer to mrbc_printf
  @param  ch	output character (ASCII)
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__
int _mrbc_printf_char(mrbc_printf *pf, int ch)
{
    if (pf->fmt.minus) {
        if (pf->p == pf->end) return -1;
        *pf->p++ = ch;
    }

    int width = pf->fmt.width;
    while (--width > 0) {
        if (pf->p == pf->end) return -1;
        *pf->p++ = ' ';
    }
    if (!pf->fmt.minus) {
        if (pf->p == pf->end) return -1;
        *pf->p++ = ch;
    }

    return 0;
}

//================================================================
/*! sprintf subcontract function for byte array.

  @param  pf	pointer to mrbc_printf.
  @param  str	pointer to byte array.
  @param  len	byte length.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__
int _mrbc_printf_bstr(mrbc_printf *pf, const char *str, int len, int pad)
{
    int ret = 0;

    if (str == NULL) {
        str = "(null)";
        len = 6;
    }
    if (pf->fmt.prec && len > pf->fmt.prec) len = pf->fmt.prec;

    int tw = len;
    if (pf->fmt.width > len) tw = pf->fmt.width;

    int remain = pf->end - pf->p;
    if (len > remain) {
        len = remain;
        ret = -1;
    }
    if (tw > remain) {
        tw = remain;
        ret = -1;
    }

    int n_pad = tw - len;

    if (!pf->fmt.minus) {
        while (n_pad-- > 0) {
            *pf->p++ = pad;
        }
    }
    while (len-- > 0) {
        *pf->p++ = *str++;
    }
    while (n_pad-- > 0) {
        *pf->p++ = pad;
    }
    return ret;
}

#if MRBC_USE_FLOAT
//================================================================
/*! sprintf subcontract function for float(double) '%f'

  @param  pf	pointer to mrbc_printf.
  @param  value	output value.
  @retval 0	done.
  @retval -1	buffer full.
*/
__GURU__
int _mrbc_printf_float(mrbc_printf *pf, double value)
{
    char fstr[16];
    const char *p1 = pf->fstr;
    char *p2 = fstr + sizeof(fstr) - 1;

    *p2 = '\0';
    while ((*--p2 = *--p1) != '%');

    //snprintf(pf->p, (pf->buf_end - pf->p + 1), p2, value);

    while (*pf->p != '\0')
        pf->p++;

    return -(pf->p == pf->end);
}
#endif

//================================================================
/*! sprintf subcontract function for char '%s'

  @param  pf	pointer to mrbc_printf.
  @param  str	output string.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__
int _mrbc_printf_str(mrbc_printf *pf, const char *str, int pad)
{
    return _mrbc_printf_bstr(pf, str, STRLEN(str), pad);
}

//================================================================
/*! sprintf subcontract function for integer '%d' '%x' '%b'

  @param  pf	pointer to mrbc_printf.
  @param  value	output value.
  @param  base	n base.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__
int _mrbc_printf_int(mrbc_printf *pf, mrbc_int value, int base)
{
    int sign = 0;
    uint32_t v = value;			// (note) Change this when supporting 64 bit.

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

    int bias_a = (pf->fmt.type == 'X') ? 'A' - 10 : 'a' - 10;

    // create string to local buffer
    char buf[64+2];				// int64 + terminate + 1
    char *p = buf + sizeof(buf) - 1;
    *p = '\0';
    do {
        int i = v % base;
        *--p = (i < 10)? i + '0' : i + bias_a;
        v /= base;
    } while (v != 0);

    // decide pad character and output sign character
    int pad;
    if (pf->fmt.zero) {
        pad = '0';
        if (sign) {
            *pf->p++ = sign;
            if (pf->p >= pf->end) return -1;
            pf->fmt.width--;
        }
    } else {
        pad = ' ';
        if (sign) *--p = sign;
    }
    return _mrbc_printf_str(pf, p, pad);
}

//================================================================
/*! sprintf subcontract function

  @param  pf	pointer to mrbc_printf
  @retval 0	(format string) done.
  @retval 1	found a format identifier.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__
int _mrbc_printf_next(mrbc_printf *pf)
{
    int ch = -1;
    pf->fmt = (mrbc_print_fmt){0};

    while (pf->p < pf->end && (ch = *pf->fstr) != '\0') {
        pf->fstr++;
        if (ch == '%') {
            if (*pf->fstr == '%') {	// is "%%"
                pf->fstr++;
            } else {
                goto PARSE_FLAG;
            }
        }
        *pf->p++ = ch;
#if 0		// no auto buffer resize, to prevent memory leak
        if (pf->p >= pf->end) {
            _mrbc_printf_resize(pf);
        }
#endif
    }
    return -(pf->p < pf->end && ch != '\0');

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.prec] type
    //   e.g. "%05d"
    while ((ch = *pf->fstr)) {
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
    while ((ch = *pf->fstr - '0'), (0 <= ch && ch <= 9)) {	// isdigit()
        pf->fmt.width = pf->fmt.width * 10 + ch;
        pf->fstr++;
    }
    if (*pf->fstr == '.') {
        pf->fstr++;
        while ((ch = *pf->fstr - '0'), (0 <= ch && ch <= 9)) {
            pf->fmt.prec = pf->fmt.prec * 10 + ch;
            pf->fstr++;
        }
    }
    if (*pf->fstr) pf->fmt.type = *pf->fstr++;

    return 1;
}

//================================================================
/*! output formatted string

  @param  fstr		format string.
*/
__GURU__
char *guru_sprintf(const char *fstr, ...)
{
    va_list ap;
    va_start(ap, fstr);

    char buf[BUF_STEP_SIZE];
    int ret = 0;
    mrbc_printf pf;

    _mrbc_printf_init(&pf, buf, BUF_STEP_SIZE, fstr);;
    
    while (ret==0 && _mrbc_printf_next(&pf)) {
    	switch(pf.fmt.type) {
        case 'c': ret = _mrbc_printf_char(&pf, va_arg(ap, int));        	 break;
        case 's': ret = _mrbc_printf_str(&pf, va_arg(ap, char *), ' '); 	 break;
        case 'd':
        case 'i':
        case 'u': ret = _mrbc_printf_int(&pf, va_arg(ap, unsigned int), 10); break;
        case 'b':
        case 'B': ret = _mrbc_printf_int(&pf, va_arg(ap, unsigned int), 2);  break;
        case 'x':
        case 'X': ret = _mrbc_printf_int(&pf, va_arg(ap, unsigned int), 16); break;
#if MRBC_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G': ret = _mrbc_printf_float(&pf, va_arg(ap, double)); 		 break;
#endif
        default:
            console_str("?format: ");
            console_char(pf.fmt.type);
            console_str("\n");
            break;
        }
    }
    va_end(ap);
    _mrbc_printf_end(&pf);

    return pf.buf;
}

__GURU__
char *guru_vprintf(const char *fstr, mrbc_value v[], int argc)		// << from c_string.cu
{
    int i   = 1;
    int ret = 1;
    mrbc_printf pf;
	char buf[BUF_STEP_SIZE];

    _mrbc_printf_init(&pf, buf, BUF_STEP_SIZE, fstr);

    while (ret==0 && _mrbc_printf_next(&pf)) {
        if (++i > argc) {						// starting from argc=2
        	console_str("ArgumentError\n");
        	return NULL;
        }

        switch(pf.fmt.type) {
        case 'c':
            if (v[i].tt==MRBC_TT_FIXNUM) {
                ret = _mrbc_printf_char(&pf, v[i].i);
            }
            break;
        case 's':
            if (v[i].tt==MRBC_TT_STRING) {
                ret = _mrbc_printf_bstr(&pf, VSTR(&v[i]), VSTRLEN(&v[i]), ' ');
            }
            else if (v[i].tt==MRBC_TT_SYMBOL) {
                ret = _mrbc_printf_str(&pf, VSYM(&v[i]), ' ');
            }
            break;
        case 'd':
        case 'i':
        case 'u':
            if (v[i].tt==MRBC_TT_FIXNUM) {
                ret = _mrbc_printf_int(&pf, v[i].i, 10);
#if MRBC_USE_FLOAT
            } else if (v[i].tt==MRBC_TT_FLOAT) {
                ret = _mrbc_printf_int(&pf, (mrbc_int)v[i].f, 10);
#endif
            } else if (v[i].tt==MRBC_TT_STRING) {
                mrbc_int ival = ATOI(VSTR(&v[i]));
                ret = _mrbc_printf_int(&pf, ival, 10);
            }
            break;
        case 'b':
        case 'B':
            if (v[i].tt==MRBC_TT_FIXNUM) {
                ret = _mrbc_printf_int(&pf, v[i].i, 2);
            }
            break;
        case 'x':
        case 'X':
            if (v[i].tt==MRBC_TT_FIXNUM) {
                ret = _mrbc_printf_int(&pf, v[i].i, 16);
            }
            break;
#if MRBC_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G':
            if (v[i].tt==MRBC_TT_FLOAT) {
                ret = _mrbc_printf_float(&pf, v[i].f);
            }
            else if (v[i].tt==MRBC_TT_FIXNUM) {
            	ret = _mrbc_printf_float(&pf, v[i].i);
            }
            break;
#endif
        default: break;
        }
    }
    _mrbc_printf_end(&pf);

    return pf.buf;		// local variable, deep copy only
}


