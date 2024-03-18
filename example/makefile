INCLUDE = `zx11-config -I` `neuz-config -I`
LIB = `zx11-config -L` `neuz-config -L`

LINK = `zx11-config -l` `neuz-config -l`
CC = gcc
CFLAGS = -Wall -ansi -O3 $(LIB) $(INCLUDE)

LINKCXX = `zx11-config -lcpp` `neuz-config -lcpp`
CXX = g++
CXXFLAGS = -Wall -std=c++17 -O3 $(LIB) $(INCLUDE)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LINK)
%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LINKCXX)
clean :
	rm -f *.o *~ core *test
