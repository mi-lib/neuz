TARGET=$(shell ls *.c | xargs -i basename {} .c | tr -s "\n" " ") $(shell ls *.cpp | xargs -i basename {} .cpp | tr -s "\n" " ")

CC = gcc
CFLAGS = -Wall -ansi -O3 $(LIB) $(INCLUDE) `zx11-config --cflags` `neuz-config --cflags`
LINK = `zx11-config -l` `neuz-config -l`

CXX = g++
CXXFLAGS = -Wall -std=c++17 -O3 $(LIB) $(INCLUDE) `zx11-config --cflags` `neuz-config --cflags`
LINKCXX = `zx11-config -lcpp` `neuz-config -lcpp`

all: $(TARGET)
%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LINK)
%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LINKCXX)
clean :
	rm -f *.o *~ core *test
