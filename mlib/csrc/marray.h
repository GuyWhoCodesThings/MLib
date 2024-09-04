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

Marray* create_marray(double* storage, int* shape, int ndim);
Marray* view_marray(Marray* marray, int* indices, int indices_length);

Marray* random_marray(double lo, double hi, int size);

Marray* elem_add_marray(Marray* marray1, Marray* marray2);

Marray* elem_mul_marray(Marray* marray1, Marray* marray2);
Marray* scale_mul_marray(Marray* marray1, double c);

Marray* matmul_marray(Marray* marray1, Marray* marray2);
Marray* reshape(Marray* marray, int* shape, int ndim);
Marray* transpose(Marray* marray);


Marray* zeros_like(Marray* marray1);
Marray* ones_like(Marray* marray1);

Marray* flatten_marray(Marray* marray);

Marray* arange_marray(int hi, int* shape, int ndim);
Marray* zeros(int* shape, int ndim);
Marray* ones(int* shape, int ndim);
Marray* eye_marray(int n, int ndim);

double get_item(Marray* marray, int* indices);

void set_item(Marray* marray, int* indices, double item);

Marray* invert_marray(Marray* marray);
int lup_decompose(Marray* marray, int N, double tol, int* P);
void lup_invert(Marray* marray, int* P, int N, Marray* inv_marray);

double sum_marray(Marray* marray);
double marray_to_item(Marray* marray);

void delete_marray(Marray* marray);
void delete_storage(Marray* marray);
void delete_strides(Marray* marray);
void delete_shape(Marray* marray);







