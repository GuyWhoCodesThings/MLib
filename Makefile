# Compiler and flags
CC = gcc
CFLAGS = -fPIC -Wall -O2

# Directories
SRC_DIR = ./mlib/csrc
LIB_DIR = ./mlib

# Source files
SRCS = $(SRC_DIR)/marray.c
OBJS = $(LIB_DIR)/marray.o

# Shared library
LIB_NAME = libmarray.so

# Target for the shared library
$(LIB_DIR)/$(LIB_NAME): $(OBJS)
	$(CC) -shared -o $@ $^

# Compile object files
$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(LIB_DIR)/$(LIB_NAME)

# Run this target to build everything
all: $(LIB_DIR)/$(LIB_NAME)

.PHONY: all clean
