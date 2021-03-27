include ../config

INCLUDE = `zx11-config -I` `neuz-config -I`
LIB = `zx11-config -L` `neuz-config -L`
LINK = -lcairo `zx11-config -l` `neuz-config -l`

CC = gcc
CFLAGS = -Wall -ansi -O3 $(LIB) $(INCLUDE)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LINK)
clean :
	rm -f *.o *~ core *test
