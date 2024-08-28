#include "marray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
Marray* create_marray(float* storage, int* shape, int ndim) {

    Marray* marray = (Marray*)safe_allocate(1, sizeof(Marray));
    
    marray->storage = storage;
    marray->shape = shape;
    marray->ndim = ndim;
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
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

// generator

Marray* elem_add_marray(Marray* marray1, Marray* marray2) {
    verify_same_shape(marray1, marray2);

    int size = marray1->size;
    int ndim = marray1->ndim;

    int* shape = (int*)safe_allocate(ndim, sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = marray1->shape[i];
    }

    float* storage = (float*)safe_allocate(size, sizeof(float));
    for (int i = 0; i < size; i++) {
        storage[i] = marray1->storage[i] + marray2->storage[i];
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

    float* storage = (float*)safe_allocate(size, sizeof(float));
    for (int i = 0; i < size; i++) {
        storage[i] = marray1->storage[i] * marray2->storage[i];
    }
    return create_marray(storage, shape, ndim);
}

Marray* scale_mul_marray(Marray* marray1, float c) {
    
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

    float* storage = (float*)malloc(size * sizeof(float));
    if (storage == NULL) {
        fprintf(stderr, "memory alloc failed\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        storage[i] = marray1->storage[i] * c;
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

    float* storage = (float*)malloc(size * sizeof(float));
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

    float* storage = (float*)malloc(size * sizeof(float));
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
    float* storage = (float*)safe_allocate(size, sizeof(float));

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            float total = 0;
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

    float* storage = (float*)malloc(size * sizeof(float));
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

Marray* squeeze_marray(Marray* marray) {

    if (marray->ndim < 2) {
        printf("must be have 2 dim");
        exit(1);
    }

    if (marray->shape[0] != 1 && marray->shape[1] != 1) {
        printf("must have 1 dim of size 1");
        exit(1);
    }

    float* storage = (float*)malloc(marray->size * sizeof(float));
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

    float* storage = (float*)malloc(marray->size * sizeof(float));
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

    float* storage = (float*)malloc(marray->size * sizeof(float));
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

// getter methods
float get_item(Marray* marray, int* indices) {

    int current_index = 0;
    for (int i = 0; i < marray->ndim; i++) {
        current_index += indices[i] * marray->strides[i];
    }
    return marray->storage[current_index];
    
}

// setter methods
float set_item(Marray* marray, int* indices, float item) {

    int current_index = 0;
    for (int i = 0; i < marray->ndim; i++) {
        current_index += i * marray->strides[i];
    }
    float prev = marray->storage[current_index];
    marray->storage[current_index] = item;
    return prev;   
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
