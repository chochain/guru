/*! @file
  @brief
  Guru bytecode loader, image built in HOST and passed into GPU
*/
#include <assert.h>
#include "vm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "guru.h"
#include "value.h"
#include "alloc.h"
#include "errorcode.h"
#include "vm.h"
#include "load.h"

#if GURU_HOST_IMAGE
//================================================================
/*!@brief
  Parse header section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of RITE header.
  @return int	zero if no error.

  <pre>
  Structure
  "RITE"	identifier
  "0004"	version
  0000		CRC
  0000_0000	total size
  "MATZ"	compiler name
  "0000"	compiler version
  </pre>
*/
__HOST__ U32
bin_to_u32(const void *s)
{
    U32 x = *((U32P)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__HOST__ U16
bin_to_u16(const void *s)
{
    U16 x = *((U16 *)s);
    return (x << 8) | (x >> 8);
}

__HOST__ int
_check_header(U8P *pos)
{
    const U8P p = *pos;

    if (memcmp(p, "RITE0004", 8) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    /* Ignore CRC */
    /* Ignore size */

    if (memcmp(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    if (memcmp(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    *pos += 22;

    return NO_ERROR;
}

//
// building memory image
//
__HOST__ void
_build_image(U8P *pirep, U8P src)
{
	guru_irep *irep = (guru_irep *)*pirep;
	U8P 	  base  = (U8P)irep;

    U32 ioff    = sizeof(guru_irep) + ((8 - sizeof(guru_irep)) & 7);
    U32 list_sz = sizeof(U32) * (irep->rlen + (irep->rlen & 1));
    U32 iseq_sz = sizeof(U32) * (irep->ilen + (irep->ilen & 1));
    U32 pool_sz = sizeof(U32) * (irep->plen + (irep->plen & 1));
    U32 sym_sz  = sizeof(U32) * (irep->slen + (irep->slen & 1));

	U8P list = U8PADD(base, ioff);
	U8P iseq = U8PADD(list, list_sz);
	U8P pool = U8PADD(iseq, iseq_sz);
	U8P sym  = U8PADD(pool, pool_sz);
	U8P tstr = U8PADD(sym, sym_sz);					// extended string table

    memcpy(iseq, U8PADD(src, irep->iseq), iseq_sz);	// copy ISEQ block
    irep->iseq = U8POFF(iseq, base);				// set iseq pointer

    U8P p = U8PADD(src, irep->pool);				// build object pool
    for (U32 i=0; i < irep->plen; i++) {
        U32  tt = *p++;
        U32  len = bin_to_u16(p);	p += sizeof(U16);
        U32  asz = len + ((8 - len) & 7);
        char buf[64+2];

        U32P v = (U32P)U8PADD(pool, i * sizeof(U32));
        switch (tt) {
        case 0: { 								// IREP_TT_STRING
            *v = U8POFF(tstr, base);
            memcpy(tstr, p, asz);				// string on 4-byte boundary
            tstr += asz;
        } break;
        case 1: { 								// IREP_TT_FIXNUM
            memcpy(buf, p, len);
            buf[len] = '\0';
            *v = atoi(buf)<<1;					// mark as number i.e. LSB=0
        } break;
#if GURU_USE_FLOAT
        case 2: { 								// IREP_TT_FLOAT
            memcpy(buf, p, len);
            buf[len] = '\0';
            *(guru_float *)v = atof(buf);
            *v |= 1;							// mark as float  i.e. LSB=1
        } break;
#endif
        default:
        	*v = 0;		// other object are not supported yet
        	break;
        }
        p += len;
    }
    irep->pool = U8POFF(pool, base);

    p = U8PADD(src, irep->sym);					// build symbol table
    for (U32 i=0; i < irep->slen; i++) {
        U32 len = bin_to_u16(p)+1;		p += sizeof(U16);	// symbol_len+'\0'
        U32 asz = len + ((8 - len) & 7);						// 8-byte alignment

        U32P v = (U32P)U8PADD(sym, i * sizeof(U32));
        *v = U8POFF(tstr, base);

        memcpy(tstr, p, asz);					// copy raw string to string table
        tstr += asz;
        p    += len;
    }
    irep->sym  = U8POFF(sym, base);
    irep->size = U8POFF(tstr, base);

    *pirep = tstr;								// return the adjusted irep pointer
}

//================================================================
/*!@brief
  read one irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL

  <pre>
  (loop n of child irep bellow)
  0000_0000	record size
  0000		n of local variable
  0000		n of register
  0000		n of child irep

  0000_0000	n of byte code  (ISEQ BLOCK)
  ...		byte codes		(padded to 4-byte align)

  0000_0000	n of pool		(POOL BLOCK)
  (loop n of pool)
  00		type
  0000		length
  ...		pool data

  0000_0000	n of symbol		(SYMS BLOCK)
  (loop n of symbol)
  0000		length
  ...		symbol data
  </pre>
*/
__HOST__ void
_get_irep_size(guru_irep *irep, U8P *pos)
{
    U8P p = *pos + 4;											// skip "IREP"

    // nlocals,nregs,rlen
    irep->nlv  = bin_to_u16(p);	p += sizeof(U16);			// number of local variables
    irep->nreg = bin_to_u16(p);	p += sizeof(U16);			// number of registers used
    irep->rlen = bin_to_u16(p);	p += sizeof(U16);			// number of child IREP blocks
    irep->ilen = bin_to_u32(p);	p += sizeof(U32);			// ISEQ (bytecodes) length

    p += U8POFF(irep, p) & 0x03;								// 32-bit align code pointer

    irep->iseq = U8POFF(p, *pos);	p += irep->ilen * sizeof(U32);	// ISEQ (code) block
    irep->plen = bin_to_u32(p);		p += sizeof(U32);				// POOL block

    irep->pool = U8POFF(p, *pos);									// pool offset
    irep->list = sizeof(guru_irep) + ((8 - sizeof(guru_irep)) & 7);	// irep list offset

    for (U32 i=0; i < irep->plen; i++) {	// scan through pool (so we know the size to allocate)
        int  tt  = *p++;
        int  len = bin_to_u16(p);   p += sizeof(U16) + len;
    }

    irep->slen = bin_to_u32(p);		p += sizeof(U32);
    irep->sym  = U8POFF(p, *pos);								// symbol offset
    for (U32 i=0; i < irep->slen; i++) {
        int len = bin_to_u16(p)+1;	p += sizeof(U16) + len;		// symbol_len+'\0'
    }
    *pos = p;								// position pointer ends here
}
//================================================================
/*!@brief
  read all irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL
*/
__HOST__ guru_irep*
_load_irep_0(U8P *pirep, U8P *pos)
{
	guru_irep *irep = (guru_irep *)*pirep;
	U8P src  = *pos;
	U8P base = (U8P)irep;

	_get_irep_size(irep, pos);			// populate metadata of current IREP
    _build_image(pirep, src);			// build IREP image in HOST, push into GPU later

    // recursively create the child irep tree
    U32P p = (U32P)U8PADD(base, irep->list);
    for (U32 i=0; i < irep->rlen; i++) {
    	guru_irep *irep_n = _load_irep_0(pirep, pos);	// a child irep
        p[i] = U8POFF(irep_n, base);					// calculate offset
    }
    return irep;
}

//================================================================
/*!@brief
  Parse IREP section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return int	zero if no error.

  <pre>
  Structure
  "IREP"	section identifier
  0000_0000	section size
  "0000"	rite version
  </pre>
*/
__HOST__ guru_irep*
_load_irep(U8P *pos)
{
    U8P p = *pos + 4;								// 4 = skip "IREP"
    U32 sec_sz = bin_to_u32(p); p += sizeof(U32);

    if (memcmp(p, "0000", 4) != 0) return NULL;		// IREP version
    p += 4;											// 4 = skip "0000"

    U32 irep_max_sz = sec_sz * 2;					// allow double the size
    U8P ibuf = (U8P)guru_malloc(irep_max_sz, 1);
    U8P ipos = ibuf;
    if (!ibuf) return NULL;							//

    memset(ibuf, 0xaa, irep_max_sz);

    guru_irep *irep = _load_irep_0(&ibuf, &p);		// recursively load irep tree

    if (irep==NULL) return NULL;					// allocation error

    U32 irep_sz = U8POFF(ibuf, ipos);
    assert(irep_sz <= irep_max_sz);					// verify allocated buffer size
    assert(sec_sz==(p - *pos));						// verify pointer arith.

    *pos += sec_sz;

    return irep;
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__HOST__ U32
_load_lvar(U8P *pos)
{
    U8P p      = *pos;
    U32 sec_sz = bin_to_u32(p+sizeof(U32));

    // TODO: local variable is not supported yet

    *pos += sec_sz;

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  vm    Pointer to VM.
  @param  ptr	Pointer to bytecode.

*/
__HOST__ void
guru_parse_bytecode(guru_vm *vm, U8P ptr)
{
    int ret = _check_header(&ptr);

    while (ret==NO_ERROR) {
        if (memcmp(ptr, "IREP", 4)==0) {
        	ret = (vm->irep = _load_irep(&ptr))==NULL
        			? LOAD_FILE_IREP_ERROR_ALLOCATION
        			: NO_ERROR;
        }
        else if (memcmp(ptr, "LVAR", 4)==0) {
            ret = _load_lvar(&ptr);
        }
        else if (memcmp(ptr, "END\0", 4)==0) {
            break;
        }
    }
}

#ifdef GURU_DEBUG
__HOST__ void
_show_irep(guru_irep *irep, U32 ioff, char level, char *idx)
{
	printf("\tirep[%c]=%c%04x: size=%d, nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			*idx, level, ioff,
			irep->size, irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	U8  *base = (U8 *)irep;
	U32 *off  = (U32 *)(base + irep->list);		// pointer to irep offset array
	for (U32 i=0; i<irep->rlen; i++) {
		*idx += 1;
		_show_irep((guru_irep *)(base + off[i]), off[i], level+1, idx);
	}
}

__HOST__ void
guru_show_irep(guru_irep *irep)
{
	char idx = 'a';
	_show_irep(irep, 0, 'A', &idx);
}
#else
__HOST__ void guru_show_irep(guru_irep *irep) {}
#endif	// GURU_DEBUG
#endif 	// GURU_HOST_IMAGE
