#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
   double* storage;
   int* shape;
   int* strides;
   int ndim;
   int size; // number of elements
   int offset;
} Marray;

#define DECOMP_TOLERANCE 0.000001f;

void print_marray_shape(Marray* marray);

Marray* create_marray(double* storage, int* shape, int ndim);
Marray* view_marray(Marray* marray, int* indices, int indices_length);

Marray* random_marray(double lo, double hi, int size);

Marray* matmul_marray(Marray* marray1, Marray* marray2);
Marray* reshape(Marray* marray, int* shape, int ndim);
Marray* transpose(Marray* marray);

double get_item(Marray* marray, int* indices);

void set_item(Marray* marray, int* indices, double item);

Marray* linespace_marray(double lo, double hi, int samples);

Marray* invert_marray(Marray* marray);
int lup_decompose(Marray* marray, int N, double tol, int* P);
void lup_invert(Marray* marray, int* P, int N, Marray* inv_marray);


double marray_to_item(Marray* marray);

void delete_marray(Marray* marray);
void delete_storage(Marray* marray);
void delete_strides(Marray* marray);
void delete_shape(Marray* marray);







