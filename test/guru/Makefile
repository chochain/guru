#
# guru  src/Makefile
#
MRBC=mrbc
SRC = \
	$(wildcard base_*.rb) \
	$(wildcard var_*.rb) \
	$(wildcard prc_*.rb) \
	$(wildcard cls_*.rb) \
	$(wildcard rng_*.rb) \
	$(wildcard str_*.rb) \
	$(wildcard ary_*.rb) \
	$(wildcard hsh_*.rb) \
	$(wildcard flt_*.rb) \
	$(wildcard bool_*.rb)

MRBS = $(SRC:.rb=.mrb)
TXTS = $(SRC:.rb=.txt)

%.mrb:%.rb
	$(MRBC) --verbose -o $@ $< | cat > $*.txt

%.txt:%.rb
	$(MRBC) --verbose $< | cat > $*.txt

all: $(MRBS) $(TXTS)

mrb: $(MRBS)

txt: $(TXTS)

clean: 
	$(RM) $(MRBS) $(TXTS) *~
