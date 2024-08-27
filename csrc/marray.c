#include "marray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// construct functions
Matrix* create_matrix(double* data, int rows, int cols) {

    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    matrix->data = data;
    matrix->rows = rows;
    matrix->cols = cols;
    return matrix;
}

Matrix* transpose(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < rows * cols; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[i * rows + j] = matrix->data[i * rows + j];
        }
        // result_data[i] = matrix->data[i];
    } 
    return create_matrix(result_data, cols, rows);
}
// 0,3,1,4,2,5
// 11 12 13   
// 21 22 23

// 11 21
// 12 22
// 13 23

void have_same_shape(Matrix* matrix1, Matrix* matrix2) {
    if (matrix1->rows != matrix2->rows) {
        printf("different number of rows, %d and %d", matrix1->rows, matrix2->rows);
        exit(1);
   }
   if (matrix1->cols != matrix2->cols) {
        printf("different number of cols, %d and %d", matrix1->cols, matrix2->cols);
        exit(1);
   }
}

// generator
Matrix* elem_add_matrix(Matrix* matrix1, Matrix* matrix2) {
    have_same_shape(matrix1, matrix2);
    int rows = matrix1->rows;
    int cols = matrix1->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[rows * i + j] = matrix1->data[rows * i + j] + matrix2->data[rows * i + j];
        }
    } 
    return create_matrix(result_data, rows, cols);
}
Matrix* elem_sub_matrix(Matrix* matrix1, Matrix* matrix2) {
    have_same_shape(matrix1, matrix2);
    int rows = matrix1->rows;
    int cols = matrix1->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[rows * i + j] = matrix1->data[rows * i + j] - matrix2->data[rows * i + j];
        }
    } 
    return create_matrix(result_data, rows, cols);
}
Matrix* elem_mul_matrix(Matrix* matrix1, Matrix* matrix2) {
    have_same_shape(matrix1, matrix2);
    int rows = matrix1->rows;
    int cols = matrix1->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[rows * i + j] = matrix1->data[rows * i + j] * matrix2->data[rows * i + j];
        }
    } 
    return create_matrix(result_data, rows, cols);
}
Matrix* elem_div_matrix(Matrix* matrix1, Matrix* matrix2) {
    have_same_shape(matrix1, matrix2);
    int rows = matrix1->rows;
    int cols = matrix1->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[rows * i + j] = matrix1->data[rows * i + j] / matrix2->data[rows * i + j];
        }
    } 
    return create_matrix(result_data, rows, cols);
}

Matrix* scal_mul_matrix(Matrix* matrix, double c) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i <  rows * cols; i++) {
        result_data[i] = matrix->data[i] * c;
    }
    return create_matrix(result_data, rows, cols);
}
Matrix* scal_add_matrix(Matrix* matrix, double c) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i <  rows * cols; i++) {
        result_data[i] = matrix->data[i] + c;
    }
    return create_matrix(result_data, rows, cols);
}
Matrix* scal_sub_matrix(Matrix* matrix, double c) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i <  rows * cols; i++) {
        result_data[i] = matrix->data[i] - c;
    }
    return create_matrix(result_data, rows, cols);
}

Matrix* scal_div_matrix(Matrix* matrix, double c) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i <  rows * cols; i++) {
        result_data[i] = matrix->data[i] / c;
    }
    return create_matrix(result_data, rows, cols);
}

Matrix* zeros(int rows, int cols) {
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i <  rows * cols; i++) {
        result_data[i] = 0;
    }
    return create_matrix(result_data, rows, cols);
}

Matrix* ones(int rows, int cols) {
    double* result_data = (double*)malloc(rows * cols * sizeof(double));
    if (result_data == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i <  rows * cols; i++) {
        result_data[i] = 1;
    }
    return create_matrix(result_data, rows, cols);
}

double max_item(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    int maximum = -INFINITY;
    for (int i = 0; i < rows * cols; i++) {
        if (maximum < matrix->data[i]) {
            maximum = matrix->data[i];
        }
    }
    return maximum;
}
double min_item(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    int minimum = -INFINITY;
    for (int i = 0; i < rows * cols; i++) {
        if (minimum > matrix->data[i]) {
            minimum = matrix->data[i];
        }
    }
    return minimum;
}

// getter methods
double get_item(Matrix* matrix, int row, int col) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    if (row > rows || col > cols) {
        fprintf(stderr, "invalid index, row in [0, %d] and col in [0, %d]", rows, cols);
        exit(1);
    }
    return matrix->data[rows * row + col];
}

// setter methods
double set_item(Matrix* matrix, int row, int col, double item) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    if (row > rows || col > cols) {
        fprintf(stderr, "invalid index, row in [0, %d] and col in [0, %d]", rows, cols);
        exit(1);
    }
    double old_item = matrix->data[rows * row + col];
    matrix->data[rows * row + col] = item;
    return old_item;
}

// garbage functions
void delete_matrix(Matrix* matrix) {
    if (matrix == NULL) return;
    free(matrix);
    matrix = NULL;
}
void delete_data(Matrix* matrix) {
    if (matrix == NULL || matrix->data == NULL) return;
    free(matrix->data);
    matrix->data = NULL;
}