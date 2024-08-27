# Compiler and flags
CC = gcc
CFLAGS = -fPIC -Wall -O2

# Directories
SRC_DIR = ./csrc
OBJ_DIR = ./obj
LIB_DIR = ./lib

# Source files
SRCS = $(SRC_DIR)/matrix.c
OBJS = $(OBJ_DIR)/matrix.o

# Shared library
LIB_NAME = libmatrix.so

# Target for the shared library
$(LIB_DIR)/$(LIB_NAME): $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) -shared -o $@ $^

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

# Run this target to build everything
all: $(LIB_DIR)/$(LIB_NAME)

.PHONY: all clean
