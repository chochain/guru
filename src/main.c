#include <stdio.h>
#include <stdlib.h>
#include "guru.h"

extern int do_cuda(void);

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

typedef struct IREP {
    uint16_t 	nlocals;   	//!< # of local variables
    uint16_t 	nregs;		//!< # of register variables
    uint16_t 	rlen;		//!< # of child IREP blocks
    uint16_t 	ilen;		//!< # of irep
    uint16_t 	plen;		//!< # of pool

    uint8_t     *code;		//!< ISEQ (code) BLOCK
    mrbc_object **pools;    //!< array of POOL objects pointer.
    uint8_t     *ptr_to_sym;
    struct IREP **reps;		//!< array of child IREP's pointer.
} mrbc_irep;

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
#if false
    for (int i = 0; i < irep->plen; i++) {
        int  tt = *p++;
        int  obj_size = _bin_to_uint16(p);	p += sizeof(uint16_t);
        char buf[MAX_OBJ_SIZE];
        mrbc_object *obj = mrbc_obj_alloc(MRBC_TT_EMPTY);
        if (obj==NULL) return NULL;

        switch (tt) {
        case 1: { // IREP_TT_FIXNUM
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';

            obj->tt = MRBC_TT_FIXNUM;
            obj->i = ATOL(buf);
        } break;
        default: break;
        }

        irep->pools[i] = obj;
        p += obj_size;
    }
#endif
    // SYMS BLOCK
    irep->ptr_to_sym = (uint8_t*)p;
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

void load_on_host(char *fname)
{
	guru_ses ses;

	int rst = input_bytecode(&ses, fname);
	mrbc_irep *irep;
	upload_bytecode(&irep, ses.req);
	dump_irep(irep);
}

extern void dump_irep(mrbc_irep *irep);
extern void dump_vm(uint8_t *vm);

int main(int argc, char **argv)
{
    //do_cuda();
	load_on_host(argv[1]);

	guru_ses ses;
	uint8_t *vm_rst = init_session(&ses, argv[1]);
	dump_vm(vm_rst);

    return 0;
}
