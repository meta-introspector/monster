# Makefile for libzkprologml

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -Iinclude
LDFLAGS = -shared

LIB_NAME = libzkprologml.so
LIB_STATIC = libzkprologml.a

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
EXAMPLE_DIR = examples

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
EXAMPLES = $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_BINS = $(EXAMPLES:$(EXAMPLE_DIR)/%.c=$(BIN_DIR)/%)

.PHONY: all clean install examples

all: $(LIB_NAME) $(LIB_STATIC)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_NAME): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(LIB_STATIC): $(OBJECTS)
	ar rcs $@ $^

examples: $(LIB_NAME) $(LIB_STATIC) | $(BIN_DIR)
	$(foreach example,$(EXAMPLES),\
		$(CC) $(CFLAGS) $(example) -L. -lzkprologml -o $(BIN_DIR)/$(notdir $(basename $(example)));)

install: $(LIB_NAME) $(LIB_STATIC)
	install -d /usr/local/lib
	install -m 755 $(LIB_NAME) /usr/local/lib/
	install -m 644 $(LIB_STATIC) /usr/local/lib/
	install -d /usr/local/include
	install -m 644 $(INC_DIR)/zkprologml.h /usr/local/include/
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_NAME) $(LIB_STATIC)

test: examples
	LD_LIBRARY_PATH=. $(BIN_DIR)/basic
