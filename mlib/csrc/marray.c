#include "marray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"


#define TOL 0.1;

// static functions
static inline double ACCESS_ELEMENT(Marray* marray, int idx) {
    return marray->storage[idx + marray->offset];
}

// todo:
// add offset into all calcuations where possible, define constant for it maybe
// figure out what to do with matmul with higher dimensions
// *MAKE sure to add prev offset when creating new view with new offset

// helper methods
void verify_same_shape(Marray* marray1, Marray* marray2) {
    if (marray1->ndim != marray2->ndim) {
        fprintf(stderr, "must have same number of dimensions\n");
        exit(1);
    }
    for (int i = 0; i < marray1->ndim; i++) {
        if (marray1->shape[i] != marray2->shape[i]) {
            fprintf(stderr, "must have same shape\n");
            exit(1);
        }
    }
}

void* safe_allocate(int size, int dtype_size) {
    void* storage = malloc(size * dtype_size);
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    return storage;
}

// construct functions
Marray* create_marray(double* storage, int* shape, int ndim) {

    Marray* marray = (Marray*)safe_allocate(1, sizeof(Marray));
    
    marray->storage = storage;
    marray->shape = shape;
    marray->ndim = ndim;
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
   
    marray->offset = 0;
    marray->size = size;
    int* strides = (int*)safe_allocate(ndim, sizeof(int));
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= marray->shape[i];
    }
    marray->strides = strides;
    return marray;
}

Marray* view_marray(Marray* marray, int* indices, int indices_length) {

    int ndim = marray->ndim;
    if (indices_length >= ndim) {
        printf("view must be an marray, use get element instead");
        exit(1);
    }
    
    int offset = 0;
    for (int i = 0; i < indices_length; i++) {
        offset += indices[i] * marray->strides[i];
    }

    int* shape = (int*)safe_allocate(ndim - indices_length, sizeof(int));
    for (int i = 0; i < ndim - indices_length; i++) {
     
        shape[i] = marray->shape[i + indices_length];
    
    }
    
    Marray* res = create_marray(marray->storage, shape, ndim - indices_length);
    res->offset = offset;
    return res;
}

// generator

Marray* elem_add_marray(Marray* marray1, Marray* marray2) {
    verify_same_shape(marray1, marray2);

    int size = marray1->size;
    int ndim = marray1->ndim;

    int* shape = (int*)safe_allocate(ndim, sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray1->shape[i];
    }

    double* storage = (double*)safe_allocate(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        storage[i] = ACCESS_ELEMENT(marray1, i) + ACCESS_ELEMENT(marray2, i);
    }
    return create_marray(storage, shape, ndim);
}

Marray* elem_mul_marray(Marray* marray1, Marray* marray2) {
    verify_same_shape(marray1, marray2);

    int size = marray1->size;
    int ndim = marray1->ndim;

    int* shape = (int*)safe_allocate(ndim, sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray1->shape[i];
    }

    double* storage = (double*)safe_allocate(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        storage[i] = ACCESS_ELEMENT(marray1, i) * ACCESS_ELEMENT(marray2, i);
    }
    return create_marray(storage, shape, ndim);
}

Marray* scale_mul_marray(Marray* marray1, double c) {
    
    int size = marray1->size;
    int ndim = marray1->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray1->shape[i];
    }

    double* storage = (double*)malloc(size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        storage[i] = ACCESS_ELEMENT(marray1, i) * c;
    }
    return create_marray(storage, shape, ndim);
}
Marray* zeros_like(Marray* marray1) {
    
    int size = marray1->size;
    int ndim = marray1->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray1->shape[i];
    }

    double* storage = (double*)malloc(size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        storage[i] = 0;
    }
    return create_marray(storage, shape, ndim);
}

Marray* ones_like(Marray* marray1) {
    
    int size = marray1->size;
    int ndim = marray1->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray1->shape[i];
    }

    double* storage = (double*)malloc(size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        storage[i] = 1;
    }
    return create_marray(storage, shape, ndim);
}

Marray* matmul_marray(Marray* marray1, Marray* marray2) {
    
    int ndim1 = marray1->ndim;
    int ndim2 = marray2->ndim;
    if (ndim1 != 2 || ndim1 != 2) {
        printf("2 dimensions needed");
        exit(1);
    }
    if (marray1->shape[ndim1 - 1] != marray2->shape[ndim2 - 2]) {
        printf("last and second to last dim must be same");
        exit(1);
    }

    int* shape = (int*)safe_allocate(2, sizeof(int));
    shape[0] = marray1->shape[0];
    shape[1] = marray2->shape[1];

    int size = shape[0] * shape[1];
    double* storage = (double*)safe_allocate(size, sizeof(double));

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            double total = 0;
            for (int k = 0; k < marray2->shape[0]; k++) {
                total += marray1->storage[i * marray1->strides[0] + k] * marray2->storage[k * marray2->strides[0] + j];
            }
            storage[i * shape[0] + j] = total;
        }
    }
    return create_marray(storage, shape, 2);
}

Marray* transpose(Marray* marray) {

    int size = marray->size;
    int ndim = marray->ndim;

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray->shape[ndim - 1 - i];
    }

    double* storage = (double*)malloc(size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    
    if (ndim == 1) {
        // vector
        for (int  i = 0; i < size; i++) {
            storage[i] = marray->storage[i];
        }
        return create_marray(storage, shape, ndim);

    }
    int index = 0;
    for (int offset = 0; offset < shape[0]; offset++) {
        for (int i = 0; i < shape[1]; i++) {
            storage[index] = marray->storage[offset + i * marray->shape[1]];
            index++;
        }
    }
    return create_marray(storage, shape, ndim);
}

Marray* reshape(Marray* marray, int* shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    if (size != marray->size) {
        printf("shape must contain same number of elements after");
        exit(1);
    }
    double* storage = (double*)safe_allocate(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        storage[i] = marray->storage[i];
    }
    return create_marray(storage, shape, ndim);
}

Marray* zeros(int* shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    double* storage = (double*)safe_allocate(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        storage[i] = 0;
    }
    return create_marray(storage, shape, ndim);
}
Marray* ones(int* shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    double* storage = (double*)safe_allocate(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        storage[i] = 1;
    }
    return create_marray(storage, shape, ndim);
}

Marray* squeeze_marray(Marray* marray) {

    if (marray->ndim < 2) {
        printf("must be have 2 dim");
        exit(1);
    }

    if (marray->shape[0] != 1 && marray->shape[1] != 1) {
        printf("must have 1 dim of size 1");
        exit(1);
    }

    double* storage = (double*)malloc(marray->size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < marray->size; i++) {
        storage[i] = marray->storage[i];
    }

    int* shape = (int*)malloc(1 * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    shape[0] = marray->shape[0] ? marray->shape[0] > marray->shape[1] : marray->shape[1];
    return create_marray(storage, shape, 1);
}

Marray* unsqueeze_marray(Marray* marray) {

    if (marray->ndim > 1) {
        printf("must be have 2 dim");
        exit(1);
    }

    double* storage = (double*)malloc(marray->size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < marray->size; i++) {
        storage[i] = marray->storage[i];
    }

    int* shape = (int*)malloc(2 * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    shape[0] = 1;
    shape[1] = marray->shape[0];
    return create_marray(storage, shape, 2);
}

Marray* flatten_marray(Marray* marray) {

    double* storage = (double*)malloc(marray->size * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < marray->size; i++) {
        storage[i] = marray->storage[i];
    }

    int* shape = (int*)malloc(2 * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    shape[0] = 1;
    for (int i = 0; i < marray->ndim; i++) {
        shape[0] *= marray->shape[i];
    }
    return create_marray(storage, shape, 1);
}

Marray* arange_marray(int hi, int* shape, int ndim) {
    double* storage = (double*)safe_allocate(hi, sizeof(double));
    double curr = 0.0;
    for (int i = 0; i < hi; i++) {
        storage[i] = curr++;
    }
    return create_marray(storage, shape, ndim);
}

Marray* eye_marray(int n, int ndim) {
    int size = n * ndim;
    int* shape = (int*)safe_allocate(ndim, sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = n;
    }
    double* storage = (double*)safe_allocate(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        storage[i] = 0;  
    }
    Marray* eye = create_marray(storage, shape, ndim);
    for (int i = 0; i < n; i++) {
        int index = 0;
        for (int j = 0; j < ndim; j++) {
            index += i * eye->strides[j];
        }
        eye->storage[index] = 1;
    }
    return eye;
}

// Marray* diagonal_marray(int , int ndim) {
//     int size = 1;
//     for (int i = 0; i < ndim; i++) {
//         size *= shape[i];
//     }
//     double* storage = (double*)safe_allocate(size, sizeof(double));
//     for (int i = 0; i < size; i++) {
//         if (i % shape[0] == 0) {
//             storage[i] = 1;
//         } else {
//             storage[i] = 0;
//         }
//     }
//     return create_marray(storage, shape, ndim);
// }

// getter methods
double get_item(Marray* marray, int* indices) {

    int current_index = 0;
    for (int i = 0; i < marray->ndim; i++) {
        current_index += indices[i] * marray->strides[i];
    }
    return ACCESS_ELEMENT(marray, current_index);
    
}

// setter methods
double set_item(Marray* marray, int* indices, double item) {

    int current_index = 0;
    for (int i = 0; i < marray->ndim; i++) {
        current_index += i * marray->strides[i];
    }
    double prev = marray->storage[current_index];
    marray->storage[current_index] = item;
    return prev;   
}

double sum_marray(Marray* marray) {
    double total = 0;
    for (int i = 0; i < marray->size; i++) {
        total += marray->storage[i];
    }
    return total;
}

double marray_to_item(Marray* marray) {
    if (marray->size != 1) {
        printf("marray size must be 1");
        exit(1);
    }
    return marray->storage[0];
}

// garbage functions
void delete_marray(Marray* marray) {
    if (marray == NULL) return;
    free(marray);
    marray = NULL;
}
void delete_storage(Marray* marray) {
    if (marray->storage == NULL) return;
    free(marray->storage);
    marray->storage = NULL;
}
void delete_strides(Marray* marray) {
    if (marray->strides == NULL) return;
    free(marray->strides);
    marray->strides = NULL;
}
void delete_shape(Marray* marray) {
    if (marray->shape == NULL) return;
    free(marray->shape);
    marray->shape = NULL;
}

Marray* invert_marray(Marray* marray) {
    double* tmp_storage = (double*)safe_allocate(marray->size, sizeof(double));
    // memcpy(tmp_storage, marray->storage, marray->size);
    for (int i = 0; i < marray->size; i++) {
        tmp_storage[i] = marray->storage[i];
    }
    Marray* tmp_marray = create_marray(tmp_storage, marray->shape, marray->ndim);
    int N = marray->shape[0];
    int* perm_arr = (int*)safe_allocate(N + 1, sizeof(int));
    int success = lup_decompose(tmp_marray, N, 0.000001, perm_arr);
    if (!success) {
        printf("inverse failed");
        exit(1);
    }
    double* inverse_storage = (double*)safe_allocate(marray->size, sizeof(double));
    int* shape = (int*)safe_allocate(marray->ndim, sizeof(int));
    // memcpy(shape, marray->shape, marray->ndim);
    for (int i = 0; i < marray->ndim; i++) {
        shape[i] = marray->shape[i];
    }
    Marray* inverse_marray = create_marray(inverse_storage, shape, marray->ndim);
    lup_invert(tmp_marray, perm_arr, N, inverse_marray);

    // clean up
    delete_strides(tmp_marray);
    delete_storage(tmp_marray);
    delete_marray(tmp_marray);
    free(perm_arr);

    return inverse_marray;
}

int lup_decompose(Marray* marray, int N, double tol, int* P) {
    double* A = marray->storage;
    int i, j, k, imax;
    double maxA, absA;

    // Initialize permutation vector P
    for (i = 0; i < N; i++) {
        P[i] = i;
    }
    P[N] = 0; // Initialize the number of row swaps

    for (i = 0; i < N; i++) {
        // Find the pivot row
        maxA = 0.0;
        imax = i;
        for (k = i; k < N; k++) {
           
            absA = fabs(A[k * N + i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        // Check for singular matrix
        // printf("curr max is %f\n", maxA);
        if (maxA < tol)
            return 0; // Failure, matrix is near singular
       

        // Pivot rows if necessary
        if (imax != i) {
            // Swap permutation
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

      
            // Swap rows in matrix A
            for (j = 0; j < N; j++) {
                double temp_val = A[i * N + j];
                A[i * N + j] = A[imax * N + j];
                A[imax * N + j] = temp_val;
            }

            // Update row swap count
            P[N]++;
        }

        // LU decomposition
        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];
            for (k = i + 1; k < N; k++) {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }

    return 1; // Decomposition done
}

void lup_invert(Marray* marray, int* P, int N, Marray* inv_marray) {
    double* A = marray->storage;     // Original matrix
    double* IA = inv_marray->storage; // Inverted matrix

    // Initialize inverse matrix with identity matrix based on permutation vector p
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i * N + j] = (P[i] == j) ? 1.0 : 0.0;
        }

        // Forward substitution to solve L * Y = I
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < i; k++) {
                IA[i * N + j] -= A[i * N + k] * IA[k * N + j];
            }
        }

        // Backward substitution to solve U * X = Y
        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++) {
                IA[i * N + j] -= A[i * N + k] * IA[k * N + j];
            }
            IA[i * N + j] /= A[i * N + i];
        }
    }
}
