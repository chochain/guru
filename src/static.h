/*! @file
  @brief
  GURU - static data declarations

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/

#ifndef GURU_SRC_STATIC_H_
#define GURU_SRC_STATIC_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ROM_CLASS		32
#define MAX_ROM_PROC		192
#define MAX_ROM_SYMBOL		128
#define MAX_ROM_STRBUF		1024

typedef struct {
	U32		sig;				// reserved for signature, versions, flags
	U16		ncls;				// total class count
	U16 	nprc;				// total proc count
	U16 	nsym;				// total symbol count
	U16		nstr;				// total byte count
	GP		cls;				// guru_class[]
	GP   	prc;				// guru_proc[]
	GP		sym;				// guru_sym[]
	GP 		str;				// U8*
} guru_rom;

extern __GURU__ guru_rom 	guru_device_rom;

#define _SYM(sid)	(((struct RSymbol*)MEMPTR(guru_device_rom.sym)) + (sid))
#define _RAW(sid)	((U8*)MEMPTR(guru_device_rom.str + _SYM(sid)->raw))

__GURU__ int	guru_rom_init();
__GURU__ void 	guru_rom_add_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n);
__GURU__ GP 	guru_rom_get_class(GT cidx);
__GURU__ GP		guru_rom_add_sym(const char *s1);		// create new symbol
__GURU__ S32	guru_rom_get_sym(U32 hash);

/*
#define ROMPTR(off)	(U8PADD(&guru_host_rom->rom, off))
#define ROM_CLS		((guru_class*)ROMPTR(0))
#define ROM_VTBL	((guru_proc*) ROMPTR(sizeof(guru_class)*MAX_ROM_CLASS))
#define ROM_SYM		((guru_symbol*)U8PADD(guru_host_rom, sym))
#define ROM_STR		(U8PADD(guru_host_rom, guru_host_rom->nstr)
*/

#ifdef __cplusplus
}
#endif
#endif
