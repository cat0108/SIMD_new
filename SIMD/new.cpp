#include<iostream>
#include<Windows.h>
#include<xmmintrin.h>
#include<emmintrin.h>
using namespace std;
alignas(16) float gdata[10000][10000];//���ж������
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];

void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j] = gdata2[i][j] = gdata3[i][j];
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
				gdata3[i][j] += gdata3[k][j];
			}
		}
	}

}

void Normal_alg(int N)
{
	int i, j, k;
	for (k = 0; k < N; k++)
	{
		for (j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (i = k + 1; i < N; i++)
		{
			for (j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}
//ֻ�Ե�һ��ѭ���Ż�
void Par_alg_part1(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata3[k][k],gdata3[k][k],gdata3[k][k],gdata3[k][k] };
		r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = _mm_loadu_ps(gdata3[k] + j);
			r1 = _mm_div_ps(r1, r0);//������������λ���
			_mm_storeu_ps(gdata3[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
		}
		gdata3[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD
		for (i = k + 1; i < n; i++)
		{
			for (j = k + 1; j < n; j++)
			{
				gdata3[i][j] = gdata3[i][j] - (gdata3[i][k] * gdata3[k][j]);
			}
			gdata3[i][k] = 0;
		}
	}
}

//ֻ������ѭ�������Ż�
void Par_alg_part(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		for (j = k + 1; j < n; j++)
		{
			gdata2[k][j] = gdata2[k][j] / gdata2[k][k];
		}
		gdata2[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[4] = { gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata2[k] + j);
				r2 = _mm_loadu_ps(gdata2[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata2[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata2[i][j] = gdata2[i][j] - (gdata2[i][k] * gdata2[k][j]);
			}
			gdata2[i][k] = 0;
		}
	}
}

//��ȫ�������Ż�
void Par_alg_all(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0=_mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j+4 <= n; j+=4)
		{
			r1 = _mm_loadu_ps(gdata[k] + j);
			r1 = _mm_div_ps(r1, r0);//������������λ���
			_mm_storeu_ps(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[4] = { gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j+=4)
			{
				r1 = _mm_loadu_ps(gdata[k] + j);
				r2 = _mm_loadu_ps(gdata[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}

void Par_alg_col(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = _mm_loadu_ps(gdata[k] + j);
			r1 = _mm_div_ps(r1, r0);//������������λ���
			_mm_storeu_ps(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD����ʱi��ʾ��
		//������Ԥ�����������Ҫ������
		float temp4[1000][4];
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			for (int m = 0; m < 4; m++)
				temp4[j / 4][m] = gdata[j + m][k];
		}
		for (i = k + 1; i < n; i++)
		{
			float temp2[4] = { gdata[k][i],gdata[k][i],gdata[k][i],gdata[k][i] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				//ȡһ�е��ĸ�Ԫ��
				float temp3[4] = { gdata[j][i],gdata[j + 1][i],gdata[j + 2][i],gdata[j + 3][i] };
				r1 = _mm_loadu_ps(temp3);
				r2 = _mm_loadu_ps(temp4[j / 4]);
				r3 = _mm_mul_ps(r0, r2);
				r1 = _mm_sub_ps(r1, r3);
				_mm_storeu_ps(temp3, r1);
				for (int m = 0; m < 3; m++)
					gdata[j + m][i] = temp3[m];
			}
			for (j; j < n; j++)
			{
				gdata[j][i] = gdata[j][i] - (gdata[k][i] * gdata[j][k]);
			}
		}
		//ȫ����Ԫ���������㣬
		for (i=k+1; i < n; i++)
			gdata[i][k] = 0.0;
	}
}


//��ά�黮��2*2
void Par_alg_twoblock(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = _mm_loadu_ps(gdata[k] + j);
			r1 = _mm_div_ps(r1, r0);//������������λ���
			_mm_storeu_ps(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD
		//��ά������������ͬʱ��Ԫ
		for (i = k + 1; i+2 <= n; i+=2)
		{
			float temp2[4] = { gdata[i][k],gdata[i][k],gdata[i+1][k],gdata[i+1][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 2 <= n; j += 2)
			{
				float temp3[4] = { gdata[k][j],gdata[k][j + 1],gdata[k][j],gdata[k][j+1] };
				r1 = _mm_loadu_ps(temp3);
				float temp4[4] = { gdata[i][j],gdata[i][j + 1],gdata[i + 1][j],gdata[i + 1][j + 1] };
				r2 = _mm_loadu_ps(temp4);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(temp4, r2);
				gdata[i][j] = temp4[0];
				gdata[i][j + 1] = temp4[1];
				gdata[i + 1][j] = temp4[2];
				gdata[i + 1][j + 1] = temp4[3];
			}
			//��ά�黮�ֿ���ʣ�����һ�к����һ��
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
		//ʣ�����һ�п���ֱ��simd�Ż�
		for (i; i < n; i++)
		{
			float temp2[4] = { gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j+=4)
			{
				r1 = _mm_loadu_ps(gdata[k] + j);
				r2 = _mm_loadu_ps(gdata[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}

int main()
{
	LARGE_INTEGER fre, begin, end;
	double gettime;
	int n;
	cin >> n;
	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Initialize(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "intial time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Normal_alg(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "normal time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_part1(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "Part1_Parallel time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_part(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "Part2_Parallel time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_col(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "All_Parallel time: " << gettime << " ms" << endl;
}
