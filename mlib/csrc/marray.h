#ifndef _CSRC_H_
#define _CSRC_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// A struct representing a multi-dimensional array
typedef struct {
   double* storage;
   int* shape;
   int* strides;
   int ndim;
   int size;
   int offset;
} Marray;

#define DECOMP_TOLERANCE 0.000001f;

// Prints the shape of the given Marray.
void print_marray_shape(Marray* marray);

// Creates a new Marray with the given storage, shape, and number of dimensions.
Marray* create_marray(double* storage, int* shape, int ndim);

// Creates a view of the given Marray using the specified indices.
Marray* view_marray(Marray* marray, int* indices, int indices_length);

// Creates a new Marray with random values between lo and hi, of the given size.
Marray* random_marray(double lo, double hi, int size);

// Performs matrix multiplication of marray1 and marray2.
Marray* matmul_marray(Marray* marray1, Marray* marray2);

// Reshapes the given Marray to the new shape with ndim dimensions.
Marray* reshape(Marray* marray, int* shape, int ndim);

// Transposes the given Marray.
Marray* transpose(Marray* marray);

// Retrieves the item at the specified indices in the Marray.
double get_item(Marray* marray, int* indices);

// Sets the item at the specified indices in the Marray to the given value.
void set_item(Marray* marray, int* indices, double item);

// Creates a new Marray with evenly spaced values from lo to hi with the given number of samples.
Marray* linespace_marray(double lo, double hi, int samples);

// Inverts the given Marray (matrix inversion).
Marray* invert_marray(Marray* marray);

// Performs LU decomposition with partial pivoting on the given Marray.
int lup_decompose(Marray* marray, int N, double tol, int* P);

// Inverts the given Marray using LU decomposition results.
void lup_invert(Marray* marray, int* P, int N, Marray* inv_marray);

// Converts a single-element Marray to a scalar value.
double marray_to_item(Marray* marray);

// Frees the memory allocated for the Marray struct.
void delete_marray(Marray* marray);

// Frees the memory allocated for the Marray's storage.
void delete_storage(Marray* marray);

// Frees the memory allocated for the Marray's strides.
void delete_strides(Marray* marray);

// Frees the memory allocated for the Marray's shape.
void delete_shape(Marray* marray);

#endif _CSRC_H_
