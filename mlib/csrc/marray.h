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
   int num_bytes; // number of bytes total
   int data_size; // size of datatype
   char* dtype; // string of datatype (ex: "int.64")
} Marray;

Marray* create_marray(double* storage, int* shape, int ndim);

Marray* elem_add_marray(Marray* marray1, Marray* marray2);

Marray* elem_mul_marray(Marray* marray1, Marray* marray2);
Marray* scale_mul_marray(Marray* marray1, float c);

Marray* matmul_marray(Marray* marray1, Marray* marray2);
Marray* reshape(Marray* marray, int* shape, int ndim);
Marray* transpose(Marray* marray);

Marray* zeros_like(Marray* marray1);
Marray* ones_like(Marray* marray1);

Marray* lu_decomp_u(Marray* marray);
Marray* lu_decomp_l(Marray* marray);
Marray* lu_inverse(Marray* lu_mat_u, Marray* lu_mat_l);

Marray* flatten_marray(Marray* marray);
Marray* squeeze_marray(Marray* marray);
Marray* unsqueeze_marray(Marray* marray);
Marray* arange_marray(int hi, int* shape, int ndim);
Marray* zeros(int* shape, int ndim);
Marray* ones(int* shape, int ndim);

double get_item(Marray* marray, int* indices);

double set_item(Marray* marray, int* indices, float item);

void delete_marray(Marray* marray);
void delete_storage(Marray* marray);
void delete_strides(Marray* marray);
void delete_shape(Marray* marray);







