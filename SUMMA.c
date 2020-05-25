/* Programa que multiplica dos matrices mediante el algoritmo SUMMA
   La matriz A es de tamaño m*k y de B es k*n
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define ROW_T 0x0
#define COL_T 0x1
#define DIM_T 0x2
#define INT_PARAMS   0x4
#define FLOAT_PARAMS 0x1
#define PACKED_BUFFER_SIZE sizeof(int)*INT_PARAMS+sizeof(float)*FLOAT_PARAMS

/* Función que multiplica una matriz por otra en la memoria local */
void prodMatrizLocal(float alfa, float *A, float *B, float *C, int m, int n, int k) {
  int i, j, l;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (l = 0; l < k; l++) {
        C[i*n+j] += alfa*A[i*k+l]*B[l*n+j];
      }
    }
  }
}

void print_order() {
	fflush(NULL);
	MPI_Barrier(MPI_COMM_WORLD);
}

/* Función que registra un datatype nuevo con el "truco" para usar scatterv */
void send_mpi_type(int rows, int cols, int stride, int size, float * matrix, float * buffer, int use_gather) {
	int elems[size];
	int posit[size];
	int ssize = round(sqrt(size));

	for (int i = 0; i < ssize; i++) {
		for (int j = 0; j < ssize; j++) {
			posit[i*ssize+j] = i*stride*rows+j*cols;
			elems[i*ssize+j] = 1;
		}
	}
	
	MPI_Datatype datatype, temp;
	MPI_Type_vector(rows, cols, stride, MPI_FLOAT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(float), &datatype);
	MPI_Type_commit(&datatype);
	
	if (use_gather) { //Esto es feo pero reutiliza un montón de código
		MPI_Gatherv(buffer, rows*cols, MPI_FLOAT, matrix, elems, posit, datatype, 0, MPI_COMM_WORLD);
	} else {
		MPI_Scatterv(matrix, elems, posit, datatype, buffer, rows*cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}	
	
	MPI_Type_free(&datatype);
	MPI_Type_free(&temp);
}

//Función auxiliar en caso de error
void end(const char * str) {
    printf("%s\n", str);
    MPI_Finalize();
    exit(0);
}

//Función de debug
void printMatrix(float * A, int ra, int ca, int rank, const char * str) {
	fflush(stdout);
	printf("PROCESO: %d Matriz: %s\n", rank, str);
	for (int i = 0; i < ra; i++) {
		for (int j = 0; j < ca; j++) {
			printf("%2.6f ", A[i*ca+j]); 
		}
		printf("\n");
	}
	printf("-----------------------------\n");
	fflush(stdout);
}

int main(int argc, char *argv[]) {
	int numprocs, rank, sqrt_numprocs;
	
	MPI_Init(&argc, &argv);
    // Determinar el rango del proceso invocado
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Determinar el numero de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	sqrt_numprocs = (int)(round(sqrt(numprocs))); //Según enunciado malla: sqrt(numprocs) x sqrt(numprocs)

	char packed_buffer[PACKED_BUFFER_SIZE];
	int packed_buffer_position = 0;
	
    int i, j, m, n, k, test;
    float alfa;

    // Proceso 0 lee parámetros de entrada
    // Parámetro 1 -> m
    // Parámetro 2 -> n	
    // Parámetro 3 -> k
    // Parámetro 4 -> alfa
    // Parámetro 5 -> booleano que nos indica si se desea imprimir matrices y vectores de entrada y salida
    if (!rank) {
        if (argc>5) {
			m    = atoi(argv[1]);
            n    = atoi(argv[2]);
			k    = atoi(argv[3]);
            alfa = atof(argv[4]);
            test = atoi(argv[5]);
        } else {
			end("NUMERO DE PARAMETROS INCORRECTO\n");
        }

        if ((m % sqrt_numprocs) || (n % sqrt_numprocs) || (k % sqrt_numprocs)) {
            end("TAMAÑO NO VALIDO, ALGÚN PARAMETRO NO ES MULTIPLO\n");    
        }
	
		//Pasamos los parametros como se pide en el enunciado en vez de multiples sends, reducimos la latencia
		MPI_Pack(&m,      1, MPI_INT,   packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, MPI_COMM_WORLD);
		MPI_Pack(&n,      1, MPI_INT,   packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, MPI_COMM_WORLD);
		MPI_Pack(&k,      1, MPI_INT,   packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, MPI_COMM_WORLD);
		MPI_Pack(&alfa,   1, MPI_INT,   packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, MPI_COMM_WORLD);
		MPI_Pack(&test,   1, MPI_FLOAT, packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, MPI_COMM_WORLD);
		
		MPI_Bcast(packed_buffer, PACKED_BUFFER_SIZE, MPI_PACKED, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(packed_buffer, PACKED_BUFFER_SIZE, MPI_PACKED, 0, MPI_COMM_WORLD);
		
		MPI_Unpack(packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, &m,    1, MPI_INT,   MPI_COMM_WORLD);
		MPI_Unpack(packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, &n,    1, MPI_INT,   MPI_COMM_WORLD);
		MPI_Unpack(packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, &k,    1, MPI_INT,   MPI_COMM_WORLD);
		MPI_Unpack(packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, &alfa, 1, MPI_FLOAT, MPI_COMM_WORLD);
		MPI_Unpack(packed_buffer, PACKED_BUFFER_SIZE, &packed_buffer_position, &test, 1, MPI_INT,   MPI_COMM_WORLD);
	}
	
	// Calculo de bloques por procesador.
    int mBloque = m/sqrt_numprocs;
	int nBloque = n/sqrt_numprocs;
	int kBloque = k/sqrt_numprocs;
		
	//Declaramos los buffers que usaremos
	float *A, *B, *C;
	float *localA = calloc(mBloque*kBloque, sizeof(float));	
    float *localB = calloc(kBloque*nBloque, sizeof(float));
    float *localC = calloc(mBloque*nBloque, sizeof(float));
	float *bufA   = calloc(mBloque*kBloque, sizeof(float));
    float *bufB   = calloc(kBloque*nBloque, sizeof(float));
	//bufC es local al algoritmo de SUMMA
	
    if(!rank){ // Proceso 0 inicializa la matriz y el vector
        A = (float *) malloc(m*k*sizeof(float));
		B = (float *) malloc(k*n*sizeof(float));
        C = (float *) malloc(m*n*sizeof(float));

        for(i=0; i<m; i++){
            for(j=0; j<k; j++){
                A[i*k+j] = 1+i+j;
            }
        }

        for(i=0; i<k; i++){
            for(j=0; j<n; j++){
                B[i*n+j] = 1+i+j;
            }
        }

        if(test){
            printf("\nMatriz A es...\n");
            for(i=0; i<m; i++){
                for(j=0; j<k; j++){
                    printf("%f ", A[i*k+j]);
                }
                printf("\n");
            }

            printf("\nMatriz B es...\n");
            for(i=0; i<k; i++){
                for(j=0; j<n; j++){
                    printf("%f ", B[i*n+j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
	 
    double t;
    MPI_Barrier(MPI_COMM_WORLD); // Barrera para garantizar una correcta medida de tiempo
    t = MPI_Wtime();
	
	//Esto me dio muchos dolores de cabeza, encapsular toda la comunicación y los tipos derivados
	//Facilitó mucho el debug
	send_mpi_type(mBloque, kBloque, k, numprocs, A, localA, 0);
	send_mpi_type(kBloque, nBloque, n, numprocs, B, localB, 0);
	
	//COMMUNICATORS
	int dimens     []        = {sqrt_numprocs, sqrt_numprocs};
	int wrap       []        = {0, 0};
	int cartesianas[]        = {0, 0};
	int topologia  [][DIM_T] = {{0, 1}, {1, 0}};
	
	MPI_Comm malla, filas, columnas;
	
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimens, wrap, 0, &malla);
	MPI_Cart_sub(malla, topologia[ROW_T], &filas);
	int key   = rank / sqrt_numprocs;
	int color = rank % sqrt_numprocs;
	
	//MPI_Cart_sub(malla, topologia[COL_T], &columnas);
	MPI_Comm_split(MPI_COMM_WORLD, color, key, &columnas); //Hacemos, como se pedia, uno por split y otro por subtopología virtual
	MPI_Cart_coords(malla, rank, 2, cartesianas);

	//SUMMA
	for (int paso = 0; paso < sqrt_numprocs; paso++) {
		float * bufC = calloc(mBloque*nBloque, sizeof(float));
		
		if (paso == cartesianas[1]) {
			for (int i = 0; i < mBloque; i++) {
				for (int j = 0; j < kBloque; j++) {
					bufA[i*kBloque+j] = localA[i*kBloque+j];
				}
			}
		}
		MPI_Bcast(bufA, mBloque*kBloque, MPI_FLOAT, paso, filas);
		
		if (paso == cartesianas[0]) {
			for (int i = 0; i < kBloque; i++) {
				for (int j = 0; j < nBloque; j++) {
					bufB[i*nBloque+j] = localB[i*nBloque+j];
				}
			}	
		}
		MPI_Bcast(bufB, kBloque*nBloque, MPI_FLOAT, paso, columnas);

        prodMatrizLocal(alfa, bufA, bufB, bufC, mBloque, nBloque, kBloque); //Reutilizado de la P2

        for(int i = 0; i < mBloque*nBloque; i++) {
            localC[i] += bufC[i];
		}
		
		free(bufC);
	}
	
	//Por último hacemos un gatherv de todo
	send_mpi_type(mBloque, nBloque, n, numprocs, C, localC, 1);
    t = MPI_Wtime()-t;

	print_order();
	
	if (test) {
		for (int i = 0; i < numprocs; i++) {
			if (i == rank) {
				printMatrix(localA, mBloque, kBloque, rank, "localA");
				printMatrix(localB, kBloque, nBloque, rank, "localB");
				printMatrix(localC, mBloque, nBloque, rank, "localC");
			}
			print_order();
		}	
	}
	
    if(!rank){
       if(test){

          printf("\nAl final matriz c es...\n");
            for(i=0; i<m; i++){
              for(j=0; j<n; j++){
                  printf("%f ", C[i*n+j]);
              }
              printf("\n");
          }
          printf("\n");

            // Solo el proceso 0 calcula el producto y compara los resultados del programa secuencial con el paralelo
            float *testC = (float *) malloc(m*n*sizeof(float));
            for(i=0; i<m; i++){
                for(j=0; j<n; j++){
                  testC[i*n+j] = 0;
                }
              }

            prodMatrizLocal(alfa, A, B, testC, m, n, k);
            int errores = 0;
			printf("Si hay algún error aparecerá aquí abajo con formato: DEBIA_SER != OBTUVIMOS\n");
            for(i=0; i<m; i++){
                for(j=0; j<n; j++){
                    if (testC[i*n+j] != C[i*n+j]) {
                      errores++;
                      printf("\n Error en la posicion (%d,%d) porque %f != %f", i, j, testC[i*m+j], C[i*m+j]);
                    }
                }
            }

            printf("\n%d errores en el producto matriz matriz con dimensiones A(%d,%d) y B(%d,%d)\n", errores, m, k, k, n);
            free(testC);
        }		

		printf("Limpiando...\n");
		MPI_Comm_free(&malla);
		MPI_Comm_free(&filas);
		MPI_Comm_free(&columnas);
		free(bufA);
		free(bufB);
		free(localA);
		free(localB);
		free(localC);
        free(A);
        free(B);
        free(C);
    }
	
    // Barrera para que no se mezcle la impresión del tiempo con la de los resultados
    print_order();
    printf("Producto matriz matriz con dimensiones A(%d,%d) y B(%d,%d) para %d procesos: Tiempo de ejecucion del proceso %3d fue %9.10lf\n", m, k, k, n, numprocs, rank, t);

    MPI_Finalize();
    return 0;
}
