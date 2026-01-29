CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lgmp

all: monster_walk_primes

monster_walk_primes: monster_walk_primes.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f monster_walk_primes

.PHONY: all clean
