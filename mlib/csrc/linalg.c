// #include "linalg.h"

// int lup_decompose(Marray* marray, int N, double tol, int* P) {
//     double* A = marray->storage;
//     int i, j, k, imax;
//     double maxA, absA;

//     // Initialize permutation vector P
//     for (i = 0; i < N; i++) {
//         P[i] = i;
//     }
//     P[N] = 0; // Initialize the number of row swaps

//     for (i = 0; i < N; i++) {
//         // Find the pivot row
//         maxA = 0.0;
//         imax = i;
//         for (k = i; k < N; k++) {
           
//             absA = fabs(A[k * N + i]);
//             if (absA > maxA) {
//                 maxA = absA;
//                 imax = k;
//             }
//         }

//         // Check for singular matrix
//         // printf("curr max is %f\n", maxA);
//         if (maxA < tol)
//             return 0; // Failure, matrix is near singular
       

//         // Pivot rows if necessary
//         if (imax != i) {
//             // Swap permutation
//             j = P[i];
//             P[i] = P[imax];
//             P[imax] = j;

      
//             // Swap rows in matrix A
//             for (j = 0; j < N; j++) {
//                 double temp_val = A[i * N + j];
//                 A[i * N + j] = A[imax * N + j];
//                 A[imax * N + j] = temp_val;
//             }

//             // Update row swap count
//             P[N]++;
//         }

//         // LU decomposition
//         for (j = i + 1; j < N; j++) {
//             A[j * N + i] /= A[i * N + i];
//             for (k = i + 1; k < N; k++) {
//                 A[j * N + k] -= A[j * N + i] * A[i * N + k];
//             }
//         }
//     }

//     return 1; // Decomposition done
// }

// void lup_invert(Marray* marray, int* P, int N, Marray* inv_marray) {
//     double* A = marray->storage;     // Original matrix
//     double* IA = inv_marray->storage; // Inverted matrix

//     // Initialize inverse matrix with identity matrix based on permutation vector p
//     for (int j = 0; j < N; j++) {
//         for (int i = 0; i < N; i++) {
//             IA[i * N + j] = (P[i] == j) ? 1.0 : 0.0;
//         }

//         // Forward substitution to solve L * Y = I
//         for (int i = 0; i < N; i++) {
//             for (int k = 0; k < i; k++) {
//                 IA[i * N + j] -= A[i * N + k] * IA[k * N + j];
//             }
//         }

//         // Backward substitution to solve U * X = Y
//         for (int i = N - 1; i >= 0; i--) {
//             for (int k = i + 1; k < N; k++) {
//                 IA[i * N + j] -= A[i * N + k] * IA[k * N + j];
//             }
//             IA[i * N + j] /= A[i * N + i];
//         }
//     }
// }