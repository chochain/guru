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
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "guru.h"

#define BUF_STEP_SIZE  80

typedef struct VPrintfFormat {
    char         type;			//!< format char. (e.g. 'd','f','x'...)
    unsigned int plus  : 1;
    unsigned int minus : 1;
    unsigned int space : 1;
    unsigned int zero  : 1;
    int          width;			//!< display width. (e.g. %10d as 10)
    int          prec;			//!< precision (e.g. %5.2f as 2)
} xprint_fmt;

typedef struct VPrintf {
    xprint_fmt 	 fmt;
    char       	 *buf;		    //!< output buffer.
    char       	 *p;		    //!< output buffer write point.
    const char 	 *end;	    	//!< output buffer end point.
    const char 	 *fstr;	        //!< format string. (e.g. "%d %03x")
} xprintf;

//================================================================
/*! initialize data container.

  @param  pf	pointer to mrbc_printf
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
  @param  fstr	format string.
*/
int _printf_init(xprintf *pf, int size, const char *fstr)
{
	char *buf = malloc(size);
	if (!buf) return -1;

    pf->buf  = pf->p = buf;
    pf->end  = buf + size - 1;		// leave one byte for '\0'
    pf->fstr = fstr;
    pf->fmt  = (xprint_fmt){0};

    return 0;
}

//================================================================
/*! clear output buffer in container.

  @param  pf	pointer to vprintf
*/
void _printf_clear(xprintf *pf)
{
    pf->p = pf->buf;		// back to head
}

//================================================================
/*! terminate ('\0') output buffer.

  @param  pf	pointer to vprintf
*/
void _printf_end(xprintf *pf)
{
	*pf->p = '\0';
}

//================================================================
/*! return string length in buffer

  @param  pf	pointer to vprintf
  @return	length
*/
int _printf_len(xprintf *pf)
{
    return pf->p - pf->buf;
}

//================================================================
/*! replace output buffer

  @param  pf	pointer to vprintf
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
*/
int _printf_adjust(xprintf *pf)
{
    int sz  = pf->end - pf->p;
	if (sz > 0) return 1;

    int off = pf->p - pf->buf;					// current offset
    int inc = BUF_STEP_SIZE;
	while ((sz + inc) < pf->fmt.width) {
		inc += BUF_STEP_SIZE;
	}

	int nsz = (pf->end - pf->buf + 1) + inc;	// new size
	char *nbuf = (char *)realloc(pf->buf, nsz);	// deep copy
	if (!nbuf) return 0;						// ENOMEM raise? TODO: leak memory.

    pf->buf = nbuf;
    pf->end = nbuf + nsz -1;					// save last byte for '\0'
    pf->p   = pf->buf + off;					// set pointer

    return 1;
}

//================================================================
/*! sprintf subcontract function for char '%c'

  @param  pf	pointer to vprintf
  @param  ch	output character (ASCII)
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
int _printf_char(xprintf *pf, int ch)
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

  @param  pf	pointer to vprintf.
  @param  str	pointer to byte array.
  @param  len	byte length.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
int _printf_bstr(xprintf *pf, const char *str, int len, int pad)
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

//================================================================
/*! sprintf subcontract function for float(double) '%f'

  @param  pf	pointer to vprintf.
  @param  value	output value.
  @retval 0	done.
  @retval -1	buffer full.
*/
int _printf_float(xprintf *pf, double value)
{
    char fstr[16];
    const char *p1 = pf->fstr;
    char *p2 = fstr + sizeof(fstr) - 1;

    *p2 = '\0';
    while ((*--p2 = *--p1) != '%');

    snprintf(pf->p, (pf->end - pf->p + 1), p2, value);

    while (*pf->p != '\0')
        pf->p++;

    return -(pf->p == pf->end);
}

//================================================================
/*! sprintf subcontract function for char '%s'

  @param  pf	pointer to vprintf.
  @param  str	output string.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
int _printf_str(xprintf *pf, const char *str, int pad)
{
    return _printf_bstr(pf, str, strlen(str), pad);
}

//================================================================
/*! sprintf subcontract function for integer '%d' '%x' '%b'

  @param  pf	pointer to vprintf.
  @param  value	output value.
  @param  base	n base.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
int _printf_int(xprintf *pf, mrbc_int value, int base)
{
    int sign = 0;
    uint32_t v = value;			// (note) Change this when supporting 64 bit.

    if (pf->fmt.type == 'd' || pf->fmt.type == 'i') {	// signed.
        if (value < 0) {
            sign = '-';
            v = -value;
        }
        else if (pf->fmt.plus)  sign = '+';
        else if (pf->fmt.space) sign = ' ';
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
    return _printf_str(pf, p, pad);
}

//================================================================
/*! sprintf subcontract function

  @param  pf	pointer to vprintf
  @retval 0	(format string) done.
  @retval 1	found a format identifier.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
int _printf_next(xprintf *pf)
{
    int ch = -1;
    pf->fmt = (xprint_fmt){0};		// overwrite the struct

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
        if (pf->p==pf->end) {
        	_printf_adjust(pf);
        }
    }
    return -(pf->p < pf->end && ch != '\0');

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.precision] type
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
char *host_sprintf(const char *fstr, ...)
{
    va_list ap;
    va_start(ap, fstr);

    int     ret = 0;
    xprintf pf;

    if (_printf_init(&pf, BUF_STEP_SIZE, fstr)) return NULL;
    
    while (ret==0 && _printf_next(&pf)) {
     	switch(pf.fmt.type) {
        case 'c': ret = _printf_char(&pf, va_arg(ap, int));        		break;
        case 's': ret = _printf_str(&pf, va_arg(ap, char *), ' '); 		break;
        case 'd':
        case 'i':
        case 'u': ret = _printf_int(&pf, va_arg(ap, unsigned int), 10); break;
        case 'b':
        case 'B': ret = _printf_int(&pf, va_arg(ap, unsigned int), 2); 	break;
        case 'x':
        case 'X': ret = _printf_int(&pf, va_arg(ap, unsigned int), 16); break;
#if MRBC_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G': ret = _printf_float(&pf, va_arg(ap, double)); 		break;
#endif
        default:
        	fprintf(stderr, "?format %c\n", pf.fmt.type);
        	break;
        }
   }
    va_end(ap);
    _printf_end(&pf);		// terminate string with '\0'

    return pf.buf;
}

