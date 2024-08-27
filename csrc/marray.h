#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
   double* data;
   int rows;
   int cols;
} Marray;

Matrix* create_matrix(double* data, int rows, int cols);

Matrix* transpose(Matrix* matrix);
Matrix* elem_add_matrix(Matrix* matrix1, Matrix* matrix2);
Matrix* elem_sub_matrix(Matrix* matrix1, Matrix* matrix2);
Matrix* elem_mul_matrix(Matrix* matrix1, Matrix* matrix2);
Matrix* elem_div_matrix(Matrix* matrix1, Matrix* matrix2);
Matrix* scal_add_matrix(Matrix* matrix, double c);
Matrix* scal_sub_matrix(Matrix* matrix, double c);
Matrix* scal_mul_matrix(Matrix* matrix, double c);
Matrix* scal_div_matrix(Matrix* matrix, double c);

double max_item(Matrix* matrix);
double min_item(Matrix* matrix);

double get_item(Matrix* matrix, int row, int col);

double set_item(Matrix* matrix, int row, int col, double item);

void delete_matrix(Matrix* matrix);
void delete_data(Matrix* matrix);








