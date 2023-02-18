#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include <windows.h>
#include <math.h>
#include <assert.h>

//定義矩陣數據型別
typedef struct
{
	double **mat;
	int m, n;
}matrix;

matrix OMP(matrix y, matrix A, int t);

//爲矩陣申請儲存空間
void initial_mat(matrix *T, int m, int n)
{
	int i;
	(*T).mat = (double**)malloc(m * sizeof(double*));
	for (i = 0; i < m; i++)
	{
		(*T).mat[i] = (double*)malloc(n * sizeof(double));
	}
	(*T).m = m;
	(*T).n = n;
}
//初始化矩陣
void initzero(matrix *T, int m, int n)
{
	int i, j;
	initial_mat(T, m, n);
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			(*T).mat[i][j] = 0;
		}
	}
}
//釋放儲存空間
void destroy(matrix *T)
{
	int i;
	for (i = 0; i < (*T).m; i++)
	{
		free((*T).mat[i]);
	}
	free((*T).mat);
}
//變換爲單位矩陣
void set_identity_matrix(matrix m) {
	int i;
	int j;
	assert(m.m == m.n);
	for (i = 0; i < m.m; ++i) {
		for (j = 0; j < m.n; ++j) {
			if (i == j) {
				m.mat[i][j] = 1.0;
			}
			else {
				m.mat[i][j] = 0.0;
			}
		}
	}
}
//矩陣轉置
void transpose_matrix(matrix input, matrix output)
{
	int i, j;
	assert(input.m == output.n);
	assert(input.n == output.m);
	for (i = 0; i < input.m; i++)
	{
		for (j = 0; j < input.n; j++)
		{
			output.mat[j][i] = input.mat[i][j];
		}
	}
}
//矩陣相乘
void multiply_matrix(matrix a, matrix b, matrix output)
{
	int i, j, k;
	assert(a.n == b.m);
	assert(output.m == a.m);
	assert(output.n == b.n);
	//printf("\n");
	for (i = 0; i < output.m; i++)
	{
		for (j = 0; j < output.n; j++)
		{
			output.mat[i][j] = 0.0;
			for (k = 0; k < a.n; k++)
			{
				//printf("a%lf b%lf", a.mat[i][k], b.mat[k][j]);
				output.mat[i][j] += a.mat[i][k] * b.mat[k][j];
			}
			//printf("%lf ", output.mat[i][j]);
		}
		//printf("\n");
	}
}
/* 交換矩陣的兩行 */
void swap_rows(matrix m, int r1, int r2) {
	double *tmp;
	assert(r1 != r2);
	tmp = m.mat[r1];
	m.mat[r1] = m.mat[r2];
	m.mat[r2] = tmp;
}
/*矩陣某行乘以一個係數  */
void scale_row(matrix m, int r, double scalar) {
	int i;
	assert(scalar != 0.0);
	for (i = 0; i < m.n; ++i) {
		m.mat[r][i] *= scalar;
	}
}

/* Add scalar * row r2 to row r1. */
void shear_row(matrix m, int r1, int r2, double scalar) {
	int i;
	assert(r1 != r2);
	for (i = 0; i < m.n; ++i) {
		m.mat[r1][i] += scalar * m.mat[r2][i];
	}
}

//矩陣求逆
int matrix_inversion(matrix input, matrix output)
{
	int i, j, r;
	double scalar, shear_needed;
	assert(input.m == input.n);
	assert(input.m == output.m);
	assert(input.m == output.n);

	set_identity_matrix(output);

	/* Convert input to the identity matrix via elementary row operations.
	   The ith pass through this loop turns the element at i,i to a 1
	   and turns all other elements in column i to a 0. */

	for (i = 0; i < input.m; ++i) {

		if (input.mat[i][i] == 0.0) {
			/* We must swap m to get a nonzero diagonal element. */

			for (r = i + 1; r < input.m; ++r) {
				if (input.mat[r][i] != 0.0) {
					break;
				}
			}
			if (r == input.m) {
				/* Every remaining element in this column is zero, so this
				   matrix cannot be inverted. */
				return 0;
			}
			swap_rows(input, i, r);
			swap_rows(output, i, r);
		}

		/* Scale this row to ensure a 1 along the diagonal.
		   We might need to worry about overflow from a huge scalar here. */
		scalar = 1.0 / input.mat[i][i];
		scale_row(input, i, scalar);
		scale_row(output, i, scalar);

		/* Zero out the other elements in this column. */
		for (j = 0; j < input.m; ++j) {
			if (i == j) {
				continue;
			}
			shear_needed = -input.mat[j][i];
			shear_row(input, j, i, shear_needed);
			shear_row(output, j, i, shear_needed);
		}
	}
	return 1;
}
matrix OMP(matrix y, matrix A, int t)
{
	int M = A.m = y.m;
	int N = A.n;
	matrix s;
	initzero(&s, N, 1);
	matrix At;
	initzero(&At, M, t);
	matrix Pos_s;
	initzero(&Pos_s, 1, t);
	matrix r_n;
	initzero(&r_n, M, 1);
	//printf("\nr_n列向量：\n");
	for (int i = 0; i < M; i++)
	{
		r_n.mat[i][0] = y.mat[i][0];
		//printf("%lf ", r_n.mat[i][0]);
	}
	matrix s_ls;
	initzero(&s_ls, t, 1);
	for (int d = 0; d < t; d++)
	{
		matrix A_T;
		initzero(&A_T, N, M);
		transpose_matrix(A, A_T);
		matrix product;
		initzero(&product, N, 1);
		multiply_matrix(A_T, r_n, product);
		/*printf("\n product列向量：\n");
		for (int i = 0; i < N; i++)
		{
			printf("%lf ", product.mat[i][0]);
		}*/
		int pos = 0;
		double max = fabs(product.mat[0][0]);
		for (int i = 1; i < N; i++)
		{
			if (max < fabs(product.mat[i][0]))
			{
				max = fabs(product.mat[i][0]);
				pos = i;
			}
		}//printf("\n pos：%d\n",pos);
		matrix Atd;
		initzero(&Atd, M, d+1);
		for (int i = 0; i < M; i++)
		{
			Atd.mat[i][d] = A.mat[i][pos];
		}
		Pos_s.mat[0][d] = pos;
		for (int i = 0; i < M; i++)
		{
			A.mat[i][pos] = 0;
		}
		matrix Atd_T;
		initzero(&Atd_T, d+1, M);
		transpose_matrix(Atd, Atd_T);
		matrix temp1;
		initzero(&temp1, d+1, d+1);
		multiply_matrix(Atd_T, Atd, temp1);
		/*printf("\n乘積：\n");
		for (int i = 0; i < d+1; i++)
		{
			for (int j = 0; j < d+1; j++)
			{
				printf("%lf ", temp1.mat[i][j]);
			}
		}*/
		matrix temp2;
		initzero(&temp2, d+1, d+1);
		matrix_inversion(temp1, temp2);
		/*printf("\n求逆：\n");
		for (int i = 0; i < d+1; i++)
		{
			for (int j = 0; j < d+1; j++)
			{
				printf("%lf ", temp2.mat[i][j]);
			}
		}*/
		matrix temp3;
		initzero(&temp3, d+1, M);
		multiply_matrix(temp2, Atd_T, temp3);
		/*printf("\n乘ATD_T：\n");
		for (int i = 0; i < d + 1; i++)
		{
			for (int j = 0; j < M; j++)
			{
				printf("%lf ", temp3.mat[i][j]);
			}
		}*/
		matrix s_ls_d;
		initzero(&s_ls_d, d + 1, 1);
		multiply_matrix(temp3, y, s_ls_d);
		/*printf("\ns：\n");
		for (int i = 0; i < d + 1; i++)
		{
			for (int j = 0; j < 1; j++)
			{
				printf("%lf ", s_ls_d.mat[i][j]);
			}
		}*/
		for (int i = 0; i < d + 1; i++)
		{
			s_ls.mat[i][0] = s_ls_d.mat[i][0];
		}
		matrix temp4;
		initzero(&temp4, M, 1);
		multiply_matrix(Atd, s_ls_d, temp4);
		for (int i = 0; i < M; i++)
		{
			r_n.mat[i][0] = y.mat[i][0] - temp4.mat[i][0];
		}
	}
	/*printf("\ns_ls:\n");
	for (int i = 0; i < t; i++)
	{
		printf("%lf ", s_ls.mat[i][0]);
	}*/
	for (int i = 0; i < t; i++)
	{
		int index = Pos_s.mat[0][i];
		//printf("[%d]%lf ", index, Pos_s.mat[0][i]);
		//printf("\n");
		s.mat[index][0] = s_ls.mat[i][0];
		//printf("[%d]%lf ",index, s_ls.mat[i][0]);
	}
	return s;
}
void main()
{
	matrix A;
	initzero(&A, 2, 5);
	A.mat[0][0] = 0.0591; A.mat[0][1] = -1.6258; A.mat[0][2] = 2.6052; A.mat[0][3] = 0.2570; A.mat[0][4] = -1.1464; 
	A.mat[1][0] = -1.4669; A.mat[1][1] = -1.9648; A.mat[1][2] = 0.9724; A.mat[1][3] = -0.9742; A.mat[1][4] = 0.5476; 
	printf("感測矩陣A:\n");
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			printf("%lf ", A.mat[i][j]);
		}printf("\n");
	}
	matrix y;
	initzero(&y, 2, 1);
	y.mat[0][0] = 7.9498;
	y.mat[1][0] = 2.9672;
	printf("\n觀測值y:\n");
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			printf("%lf ", y.mat[i][j]);
		}printf("\n");
	}
	int t = 1;
	matrix s;
	initzero(&s, 5, 1);
	s = OMP(y, A, t);
	matrix PSi;
	initzero(&PSi, 5, 5);
	for (int i = 0; i < 5; i++)
	{
		PSi.mat[i][i] = 1;
	}
	printf("\n稀疏基PSi:\n");
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j <5; j++)
		{
			printf("%lf ", PSi.mat[i][j]);
		}printf("\n");
	}
	matrix x_r;
	initzero(&x_r, 5, 1);
	multiply_matrix(PSi, s, x_r);

	printf("\ns:\n");
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", s.mat[i][0]);
	}
	matrix x;
	initzero(&x, 5, 1);
	x.mat[2][0] = 3.0515;
	printf("\n原始信號x:\n");
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", x_r.mat[i][0]);
	}

	printf("\n恢復信號x_r:\n");
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", x_r.mat[i][0]);
	}

	getchar();
}