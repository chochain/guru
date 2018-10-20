/*! @file
  @brief
  If you want to add your own extension,
  add your code in c_ext.c and c_ext.h. 

  <pre>
  Copyright (C) 2015 Kyushu Institute of Technology.
  Copyright (C) 2015 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <stdio.h>
#include <stdlib.h>
#include "opcode.h"
#include "c_ext.h"

uint32_t bin_to_uint32(const void *s)
{
    uint32_t x = *((uint32_t *)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

uint16_t bin_to_uint16(const void *s)
{
    uint16_t x = *((uint16_t *)s);
    return (x << 8) | (x >> 8);
}

void _memcpy(const uint8_t *d, const uint8_t *s, size_t sz)
{
    for (int i=0; i<sz; i++, *d++==*s++);
}

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

    if (_memcmp(p + 14, "MATZ0000", 8) != 0) {
        return -1;
    }
    *pos += 22;
    return 0;
}

int load_irep_1(mrbc_irep *irep, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;			// skip "IREP"

    // nlocals,nregs,rlen
    irep->nlocals = bin_to_uint16(p);	p += sizeof(uint16_t);
    irep->nregs   = bin_to_uint16(p);	p += sizeof(uint16_t);
    irep->rlen    = bin_to_uint16(p);	p += sizeof(uint16_t);
    irep->ilen    = bin_to_uint32(p);	p += sizeof(uint32_t);
    irep->code    = (uint8_t *)p;       p += irep->ilen * sizeof(uint32_t);		// ISEQ (code) block
    irep->plen    = bin_to_uint32(p);	p += sizeof(uint32_t);					// POOL block

    // allocate memory for child irep's pointers
    if (irep->rlen) {
        irep->reps = (mrbc_irep **)malloc(sizeof(mrbc_irep *) * irep->rlen);
        if (irep->reps==NULL) return -1;
    }
    if (irep->plen) {
        irep->pools = (mrbc_object**)malloc(sizeof(void*) *irep->plen);
        if (irep->pools==NULL) return -1;
    }

#define MAX_OBJ_SIZE 100

    for (int i = 0; i < irep->plen; i++) {
        int  tt = *p++;
        int  obj_size = bin_to_uint16(p);	p += sizeof(uint16_t);

        char buf[MAX_OBJ_SIZE];

        mrbc_object *obj = (mrbc_object *)malloc(sizeof(mrbc_object));
        if (obj==NULL) return -1;

        switch (tt) {
        case 1: { 			// IREP_TT_FIXNUM
            _memcpy((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';

            obj->tt = MRBC_TT_FIXNUM;
            obj->i  = atol(buf);
        } break;
        default:
        	obj->tt = MRBC_TT_EMPTY;
        	break;
        }

        irep->pools[i] = obj;
        p += obj_size;
    }
    // SYMS BLOCK
    irep->sym = (uint8_t*)p;
    int sym_cnt = bin_to_uint32(p);		p += sizeof(uint32_t);
    while (--sym_cnt >= 0) {
        int len = bin_to_uint16(p);
        p += sizeof(uint16_t)+len+1;	// symbol length+'\0'
    }
    *pos = p;

    return 0;
}

//================================================================
/*!@brief
  read all irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL
*/
mrbc_irep *load_irep_0(const uint8_t **pos)
{
    // new irep
    mrbc_irep *irep = (mrbc_irep *)malloc(sizeof(mrbc_irep));

    if (irep==NULL) return NULL;

    int ret = load_irep_1(irep, pos);
    if (ret!=0) return NULL;

    int i;
    for (i = 0; i < irep->rlen; i++) {
        irep->reps[i] = load_irep_0(pos);
    }
    return irep;
}


int load_irep(mrbc_irep **pirep, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;			// 4 = skip "IREP"
    int   sec_size = bin_to_uint32(p);

    p += sizeof(uint32_t);
    if (_memcmp(p, "0000", 4) != 0) {		// rite version
        return -1;
    }
    p += 4;

    *pirep = load_irep_0(&p);
    if (*pirep==NULL) {
        return -1;
    }

    *pos += sec_size;
    return 0;
}

int load_lvar(const uint8_t **pos)
{
    const uint8_t *p = *pos;

    /* size */
    *pos += bin_to_uint32(p+sizeof(uint32_t));

    return 0;
}

void upload_bytecode(mrbc_irep **pirep, const uint8_t *ptr)
{
    int ret;

    ret = load_header(&ptr);
    while (ret==0) {
        if (_memcmp(ptr, "IREP", 4)==0) {
            ret = load_irep(pirep, &ptr);
        }
        else if (_memcmp(ptr, "LVAR", 4)==0) {
            ret = load_lvar(&ptr);
        }
        else if (_memcmp(ptr, "END\0", 4)==0) {
            break;
        }
    }
    return;
}

int input_bytecode(guru_ses *ses, const char *rite_fname)
{
  FILE *fp = fopen(rite_fname, "rb");

  if (fp==NULL) {
    fprintf(stderr, "File not found\n");
    return -1;
  }

  // get filesize
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  ses->req = (uint8_t *)malloc(sz);

  if (ses->req==NULL) {
	  fprintf(stderr, "memory allocate error\n");
	  return -1;
  }
  else {
	  fread(ses->req, sizeof(uint8_t), sz, fp);
  }
  fclose(fp);

  return 0;
}

void load_on_host(mrbc_vm *vm, char *fname)
{
	guru_ses ses;

	int rst = input_bytecode(&ses, fname);
	upload_bytecode(&(vm->irep), ses.req);
}

void dump_irep(mrbc_irep *irep);

void guru_init_ext(mrbc_vm *vm, char *fname)
{
	load_on_host(vm, fname);
	dump_irep(vm->irep);
}
