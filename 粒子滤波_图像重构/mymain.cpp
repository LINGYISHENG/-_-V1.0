#pragma comment (lib, "libgsl.a")
/* From GSL */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
/* From opencv*/
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
/* std */
#include<iostream>
#include<stdlib.h>
#include<stdio.h> 
#include<time.h>

using namespace std;
using namespace cv;

#define NEW_PDF //ȡ������ʱֱ��ʹ�����Ƶ�λ����Ϊת��ģ��
#define ADD_NOISE //������
//#define FILE_OUT //�ļ������·�����·�����������



/************����*************/
constexpr auto IMAGE_FILE_NAME = "fprint512.png"; //lena.tiff    timg.jpg    building.jpg     niose building.png  noise building16.png  barbara.png boat.png  fprint3.png
constexpr auto OUTPUT_PATH = "C:\\Users\\15876\\Desktop\\result\\fprint512";
constexpr auto OUTPUT_IMAGE_FILE_NAME = "\\fprint512_nosie16_result.png";
constexpr auto OUTPUT_PATH_Noiseimg = "C:\\Users\\15876\\Desktop\\result\\fprint512";
constexpr auto OUTPUT_NOISE_IMAGE_FILE_NAME = "\\fprint512_noise16.png";
constexpr auto NUM_PARTICLE = 60;//��������
constexpr auto NUM_PDF = 10;
constexpr auto NUM_PDF2 = 10;
constexpr auto T = 6;//��������
constexpr auto transS = 17;//ת��ģ�ͳ߶�s
constexpr auto msS = 7;//���㷽��ͱ�׼��ĳ߶�
constexpr auto NUM_COL = 512;
constexpr auto NUM_ROW = 512;//ͼƬ��С��void meanstdvalue_img���õ������ͼƬ��С����һ��

static double meanMat[NUM_COL][NUM_ROW] = { 0 };
static double stdMat[NUM_COL][NUM_ROW] = { 0 };//����ÿ������Χ����ľ�ֵ�ͱ�׼��


/*********���ӽṹ��************/
typedef struct particle {
	int x;			// ��ǰx����
	int y;			// ��ǰy����
	double mean;        //������Χ�ֲ������ֵ
	double var;         //������Χ�ֲ����򷽲�
	double weight;		// �����ӵ�Ȩ��
	int xHistory[T];
	int yHistory[T];   //λ�ü�¼
	double intensityHistory[T]; //������λ�õ��������
	//double weightHistory[T];//Ȩ�ؼ�¼


} PARTICLE;


/*���ֵ�ͱ�׼��*/
double meanstdvalue(Mat scr, int col, int row, int windowsize)
{
	Mat roi = scr(Rect(col-(windowsize-1)/2, row-(windowsize-1)/2, windowsize, windowsize));
	Mat tmp_m, tmp_sd;
	meanStdDev(roi, tmp_m, tmp_sd);
	double m = tmp_m.at<double>(0, 0);
	double sd = tmp_sd.at<double>(0, 0);
	//cout << "Mean: " << m << " , StdDev: " << sd << endl;
	
	return m;
}

void meanstdvalue(Mat scr, int col, int row, int windowsize, double* mean, double* std)//��������ͼ���������������ڴ�С������ֵ�ĵ�ַ
{
	Mat roi = scr(Rect(col - (windowsize - 1) / 2, row - (windowsize - 1) / 2, windowsize, windowsize));
	Mat tmp_m, tmp_sd;
	meanStdDev(roi, tmp_m, tmp_sd);
	double m = tmp_m.at<double>(0, 0);
	double sd = tmp_sd.at<double>(0, 0);
	//cout << "Mean: " << m << " , StdDev: " << sd << endl;
	*mean = m;
	*std = sd;
}

void meanstdvalue_img(Mat scr, int windowsize)
{
	double mean, std;

	for (int i = ((windowsize - 1) / 2); i < ( NUM_ROW- (windowsize - 1) / 2); i++)
	{
		for (int j = ((windowsize - 1) / 2); j < (NUM_COL - (windowsize - 1) / 2); j++)
		{
			meanstdvalue(scr, j, i, windowsize, &mean, &std);
			meanMat[j][i] = mean;
			stdMat[j][i] = std;
		}
	}
}


/*************************���ӳ�ʼ��******************************************/
void particle_init(particle* particles, int x, int y)
{
	for (int i = 0; i < NUM_PARTICLE; i++)
	{
		//�������ӳ�ʼ������������λ��
		particles[i].x = x;
		particles[i].y = y;
		//particles[i].xPre = particles[i].x;
		//particles[i].yPre = particles[i].y;
		particles[i].xHistory[T] = { 0 };
		particles[i].yHistory[T] = { 0 };
		particles[i].intensityHistory[T] = { 0 };

		//Ȩ��ȫ��Ϊ0
		particles[i].weight = 0;
	}
}


//��ѡ���������ھ�ֵ�������Ƶĵ�,for��ij����
int selectpixels(Mat imgcvt, int col, int row, int windowsize, double meanthv, double stdthv, int d[NUM_PARTICLE][2])
{
	double stdX0 = 0;
	double meanX0 = 0;
	double stdx = 0;
	double meanx = 0;
	int numVector = 0;
	//����ʹ�õı���
	double meanstd[NUM_PARTICLE];
	for (int i = 0; i < NUM_PARTICLE; i++)
	{
		meanstd[i] = 10000;
	}
	double meanstdx;

	//// ��ɾ
	//Mat img;
	//cvtColor(imgcvt, img, CV_GRAY2RGB);

	//windowsize = 41;
	//meanthv = 3;
	//stdthv = 1.5;
	//meanstdvalue(imgcvt, col, row, 7, &stdX0, &meanX0);//Ч�ʵ�ȥ�������ɱ��
	stdX0 = stdMat[col][row];
	meanX0 = meanMat[col][row];
	
	for (int i = col - (windowsize - 1) / 2; i <= col + (windowsize - 1) / 2; i++)
	{
		for (int j = row - (windowsize - 1) / 2; j <= row + (windowsize - 1) / 2; j++)
		{
			//meanstdvalue(imgcvt, i, j, 7, &stdx, &meanx);//Ч�ʵ�ȥ�������ɱ��
			stdx = stdMat[i][j];
			meanx = meanMat[i][j];
			if (((-meanthv) < (meanx - meanX0)) && ((meanx - meanX0) < meanthv) && ((-stdthv) < (stdx - stdX0)) && ((stdx - stdX0) < stdthv))
			{
				numVector++;
				//ѡ������Np��������ת��ģ��,����ʹ�þ�ֵ�ӱ�׼��ֱ�*��ֵ�ĵ���
				meanstdx = meanx / meanthv + stdx / stdthv;
				for (int a = (NUM_PARTICLE - 1); a >=0; a--)
				{
					if (meanstd[a] > meanstdx)
					{
						if (a != 0)
						{
							if (a >= NUM_PARTICLE - 1)
							{ }
							else
							{
								meanstd[a + 1] = meanstd[a];
								d[a + 1][0] = d[a][0];
								d[a + 1][1] = d[a][1];
							}
						}
						else 
						{
							meanstd[a + 1] = meanstdx;
							d[a + 1][0] = i - col;
							d[a + 1][1] = j - row;
							break;
						}
					}
					else 
					{
						if (a >= NUM_PARTICLE - 1)
						{
							break;
						}
						else 
						{
							meanstd[a + 1] = meanstdx;
							d[a + 1][0] = i-col;
							d[a + 1][1] = j-row;
							break;
						}
					}
				}

				//��ʾ���ֿ���ɾ��
				//��������Ϳ��
				
				//img.at<Vec3b>(j, i)[1] = 0;
				//img.at<Vec3b>(j, i)[0] = 0;
				//img.at<Vec3b>(j, i)[2] = 255;
			}

		}
	}
	/*if (numVector < 60)
	{

	}*/
	////��ʾ����,����ɾ��
	////������ĵ㣬��ɫ
	//img.at<Vec3b>(row, col)[1] = 255;
	//img.at<Vec3b>(row, col)[0] = 0;
	//img.at<Vec3b>(row, col)[2] = 0;
	////��ʾ�������ص�
	//Mat jubu = img(Rect(col - ((windowsize - 1) / 2), row - ((windowsize - 1) / 2), 40, 40));
	//namedWindow("jubu", 0);
	//cvResizeWindow("jubu", 512, 512);
	//imshow("jubu", img);
	//waitKey(0);

	return MIN(60, numVector);;
}

//k���ڷ������ܶȹ���
int transition_modeling(int scale, int d[][2], int sized, double transModel[transS][transS])
{
	double dist[transS][transS][60] = { 0 };
	double dist_sort[transS][transS][60] = { 0 };
	double x, y;
	int neighborK = scale;
	int Normal_d[60][2] = { 0 };
	for (int i = 0; i < sized; i++)
	{
		Normal_d[i][0] = d[i][0] + (transS - 1) / 2;
		Normal_d[i][1] = d[i][1] + (transS - 1) / 2;
	}
	//����������ÿ���㵽ÿ�������ľ��룬�����򷽱�Ѱ��K�������������
	for (int i = 0; i < transS; i++)
	{
		for (int j = 0; j < transS; j++)
		{
			for (int k = 0; k < sized; k++)
			{
				x = i - Normal_d[k][0];
				y = j - Normal_d[k][1];
				dist[i][j][k] = sqrt(pow(x, 2) + pow(y, 2));
			}
			sort(&dist[i][j][0], &dist[i][j][sized - 1] + 1);
		}
	}
	//K���ڣ�����ÿ�����صĸ���k/n/v
	double sum = 0;
	double max = 0;
	for (int i = 0; i < transS; i++)
	{
		for (int j = 0; j < transS; j++)
		{
			double k = neighborK;
			double n = sized;
			double d= dist[i][j][neighborK];
			double v = (3.141593*pow(d, 2));
			transModel[i][j] = k / n / v;
			sum += transModel[i][j];
		}
	}
	//��һ��
	for (int i = 0; i < transS; i++)
	{
		for (int j = 0; j < transS; j++)
		{
			transModel[i][j] = transModel[i][j] / sum;
			if (max < transModel[i][j])
				max = transModel[i][j];
		}
	}

	return 1;
}


/***********����״̬ת��**********/
//��ض���
/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 0.05
#define TRANS_Y_STD 0.05
/* autoregressive dynamics parameters for transition model */
#define A1  2.0//2.0
#define A2  -1.0//-1.0
#define B0  10
//����������λ��dΪת��ģ��ʱ��ת�ƺ�����NUM_PDF��ͬ����ʽ�����в�ʹ��
particle transition_FirstTime(particle p, gsl_rng* rng, int d[NUM_PARTICLE][2],int H,int W)
{
	int x, y,dRANK;
	particle pn=p;

	/* sample new state using second-order autoregressive dynamics */
	dRANK = (int)gsl_ran_flat(rng, 0, NUM_PDF);
	x = p.x + d[dRANK][0];// +B0 * gsl_ran_gaussian(rng, TRANS_X_STD);
	pn.x = MAX(4.0, MIN((double)W - 4.0, x));
	y = p.y + d[dRANK][1];// +B0 * gsl_ran_gaussian(rng, TRANS_Y_STD);
	pn.y = MAX(4.0, MIN((double)H - 4.0, y));

	//pn.xPre = p.x;
	//pn.yPre = p.y;
	pn.weight = 0;

	return pn;
}

//����������λ��dΪת��ģ��ʱ��ת�ƺ�������ʽ�����в�ʹ��
particle transition(particle p, gsl_rng* rng, int d[NUM_PARTICLE][2], int H, int W)
{
	int x, y,dRANK;
	particle pn = p;

	/* sample new state using second-order autoregressive dynamics */
	dRANK = (int)gsl_ran_flat(rng, 0, NUM_PDF2);
	x = p.x + d[dRANK][0];// +B0 * gsl_ran_gaussian(rng, TRANS_X_STD);
	pn.x = MAX(4.0, MIN((double)W - 4.0, x));
	y = p.y + d[dRANK][1];// +B0 * gsl_ran_gaussian(rng, TRANS_Y_STD);
	pn.y = MAX(4.0, MIN((double)H - 4.0, y));

	//pn.xPre = p.x;
	//pn.yPre = p.y;
	pn.weight = 0;

	return pn;
}

//���ݸ����ܶȺ������������
particle transition_new(particle p, gsl_rng* rng, double transmodel[transS][transS], int H, int W)//����˵�������ӣ���������ӣ�ת��ģ�ͣ�ͼ���ͼ���
{
	int x, y;
	double u;
	double L_model[transS*transS] = { 0 };
	double cdf[transS*transS] = { 0 };
	particle pn = p;

	//���������
	u = gsl_ran_flat(rng, 0, 1);
	//�۳�һά�ģ�һ��һ�У�����ʵ���Բ��ò����﷽�㴦��������Ч�ʣ�
	for (int i = 0; i < transS; i++)
	{
		for (int j = 0; j < transS; j++)
		{
			L_model[i*transS + j] = transmodel[j][i];
		}
	}
	//״̬ת��ģ�͵�cdf
	for (int i = 1; i < (transS*transS); i++)
	{
		cdf[i] = cdf[i - 1] + L_model[i];
	}

	for (int i = 0; i < (transS*transS); i++)
	{
		if (u < cdf[i])
		{
			pn.x = p.x + i % transS - (transS - 1) / 2;
			pn.y = p.y + i / transS - (transS - 1) / 2;
			break;
		}
	}

	//pn.xPre = p.x;
	//pn.yPre = p.y;
	pn.weight = 0;

	return pn;
}


//Ȩ�ع�һ��
void normalize_weights(particle* particles, int n)//����˵�������ӵ�ַ����������
{
	float sum = 0;
	int i;

	for (i = 0; i < n; i++)
		sum += particles[i].weight;
	for (i = 0; i < n; i++)
		particles[i].weight /= sum;
}

//���qsort�����Ӱ�Ȩ������
int particle_cmp(const void* p1, const void* p2)
{
	//����������qsort����������������ֵ: (1) <0ʱ��p1����p2ǰ��   (2)  >0ʱ��p1����p2����
	particle* _p1 = (particle*)p1;
	particle* _p2 = (particle*)p2;
	//������ɴ�С������
	return (int)((_p2->weight - _p1->weight)*1000000);//�������Ŀ���������û�ģ�
}
 
//���ַ��������д��ڸ���ֵ����Сֵ����
int get_min_index(double *array, int length, double _value)
{
	int _index = (length - 1) / 2;
	int last_index = length - 1;
	int _index_up_limit = length - 1;
	int _index_down_limit = 0;
	//���жϼ�ֵ
	if (array[0] <= _value) {
		return 0;
	}
	if (array[length - 1] > _value) {
		return length - 1;
	}
	for (; _index != last_index;) {
		//cout << _index << endl;
		last_index = _index;
		if (array[_index] > _value) {
			_index = (_index_up_limit + _index) / 2;
			_index_down_limit = last_index;
		}
		else if (array[_index] < _value) {
			_index = (_index_down_limit + _index) / 2;
			_index_up_limit = last_index;
		}
		else if (array[_index] == _value) {
			_index--;
			break;
		}
	}
	//cout << "final result:" << endl;
	//cout << _index << endl;
	return _index;
}

//�ز���
void resample(particle* particles, particle* new_particles, int num_particles)
{
	//����ÿ�����ӵĸ����ۼƺ�
	double sum[NUM_PARTICLE], temp_sum = 0;
	int k = 0;
	for (int j = num_particles - 1; j >= 0; j--) {
		temp_sum += particles[j].weight;
		sum[j] = temp_sum;
	}
	//Ϊÿ����������һ�����ȷֲ���0��1���������
	RNG sum_rng(time(NULL));
	double Ran[NUM_PARTICLE];
	for (int j = 0; j < num_particles; j++) {
		sum_rng = sum_rng.next();
		Ran[j] = sum_rng.uniform(0.0, 1.0);
	}
	//�����Ӹ����ۻ����������ҵ���С�Ĵ��ڸ�������������������Ƹ�����������һ�ε��µ����������� ����Ȩ�ظߵ����ӿ�ʼ��
	for (int j = 0; j < num_particles; j++) {
		int copy_index = get_min_index(sum, num_particles, Ran[j]);
		new_particles[k++] = particles[copy_index];
		if (k == num_particles)
			break;
	}
	//�������Ĳ�����ɣ����������������������ԭ������������������Ȩ����ߵ����ӣ�ֱ�����������
	while (k < num_particles)
	{
		new_particles[k++] = particles[0]; //����Ȩֵ��ߵ�����
	}
	//�����������鸲�Ǿõ���������
	for (int i = 0; i < num_particles; i++)
	{
		particles[i] = new_particles[i];  //���������ӵ�particles
	}
}

double generateGaussianNoise(double mu, double sigma)
{
	//����Сֵ
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flagΪ�ٹ����˹�������X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//�����������
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flagΪ�湹���˹�������
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//�ж�ͼ���������
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//��Ӹ�˹����
			int val = dstImage.ptr<uchar>(i)[j] + generateGaussianNoise(0, 0.5) * 16;//ԭ����*32
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}


//main
int main()
{
	Mat imgRD = imread(IMAGE_FILE_NAME);//lena.tiff    timg.jpg
	Mat imgcvt;
	cvtColor(imgRD, imgcvt, CV_RGB2GRAY);
	namedWindow("ԭʼͼ��", 2);
	cvResizeWindow("ԭʼͼ��", imgcvt.cols, imgcvt.rows);
	imshow("ԭʼͼ��", imgcvt);
	waitKey(100);

	//����������������ͼƬ
#ifdef ADD_NOISE
	imgcvt = addGaussianNoise(imgcvt);
#ifdef FILE_OUT
	imgcvt = addGaussianNoise(imgcvt);
	string output_path_noiseimg = OUTPUT_PATH_Noiseimg;
	string output_file_name_noiseimg = output_path_noiseimg.append(OUTPUT_NOISE_IMAGE_FILE_NAME);
	imwrite(output_file_name_noiseimg, imgcvt);
#endif //FILE_OUT
#endif // ADD_NOISE

	PARTICLE P[NUM_PARTICLE];//����
	PARTICLE NEW_P[NUM_PARTICLE];

	//GSL�������ʼ��
	gsl_rng* rng;
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, time(NULL));

	////��ʾ����ʱʹ��
	//Mat img;
	//cvtColor(imgcvt, img, CV_GRAY2RGB);

	//�����ͼ��
	Mat imgOut;
	imgOut = imgcvt.clone();

	//�ֲ���ֵ�ͷ�������ظ����㣬�����ڴ�
	int intensityX0;
	double  stdX0, meanX0;
	double stdx, meanx;
	int d[NUM_PARTICLE][2] = { 0 };
	static int W = imgcvt.cols;
	static int H = imgcvt.rows;
	//Ȩ�ؼ������
	double Dsk = 0;
	int SigmaG = 15;
	double Dvk = 0;
	int SigmaV = 15;
	double Wk = 0;
	int err = 0;
	//�ع�ֵ�������
	double intensitySum = 0;
	double intensityPartical = 0;
	double intensityOutSum = 0;
	int intensityOut = 0;
	//ת��ģ��
	double transmodel[transS][transS];
	int num_d = 0;

	cout << "ͼƬ�ķֱ���Ϊ�� " << W << "*" << H << endl;
	cout << "���õķֱ���Ϊ��" << NUM_COL << "*" << NUM_ROW <<"  ";
	if (NUM_COL == W && NUM_ROW == H)
		cout << endl;
	else
	{
		cout << "���õķֱ��ʴ��󣡣����޸�NUM_COL��NUM_ROW��ֵ��" << endl;
		cout << "�ع�δ����!"<< endl;
		waitKey(0);
		return -1;
	}

	//��������ͼ���������ؾֲ��ķ���;�ֵ������meanmat��stdmat��
	meanstdvalue_img(imgcvt, msS);
	
	//��������forѭ������������ͼ��
	for (int row = 30; row < (H - 30); row++)
	{
		cout << "��" << row << "��; " << endl;
		for (int col = 30; col < (W - 30); col++)
		{
			//cout << "��" << col << "��; " ;
			for (int k = 0; k < T; k++)
			{
				//ÿ�����ض�������
				intensityX0 = imgcvt.at<uchar>(row, col);
				if(k==0)
				{
					//��ʼ��λ��d��ת��ģ�ͣ������㷽��;�ֵ���ƶ�ǰ<��������>��λ��d
					for (int i = 0; i < NUM_PARTICLE; i++)
					{
						d[i][0] = 0;
						d[i][1] = 0;
					}
					//����ת��ģ��
					num_d = selectpixels(imgcvt, col, row, transS, 5, 3, d);//3,1.5
#ifdef NEW_PDF
					//���£�ת��ģ��
					transition_modeling(2, d, num_d, transmodel);
#endif // NEW_PDF

					//��ʼ��
					particle_init(P, col, row);
				}
				//��ÿ�����ӵĲ���
				////��ʾ����ɾ,��ʾÿ������
				//cvtColor(imgcvt, img, CV_GRAY2RGB);
				for (int i = 0; i < NUM_PARTICLE; i++)
				{
					//����
					if (k == 0)
					{
						//P[i] = transition_FirstTime(P[i], rng, d, H, W);
					}
					else
					{
#ifdef NEW_PDF
						P[i] = transition_new(P[i], rng, transmodel, H, W);
#else
						P[i] = transition(P[i], rng, d, H, W);
#endif // NEW_PDF
					}
					P[i].intensityHistory[k] = imgcvt.at<uchar>(P[i].y, P[i].x);
					////��ʾ����ɾ����ʾÿ������
					//img.at<Vec3b>(P[i].y, P[i].x)[1] = 0;
					//img.at<Vec3b>(P[i].y, P[i].x)[0] = 0;
					//img.at<Vec3b>(P[i].y, P[i].x)[2] = 255;

					//img.at<Vec3b>(row, col)[1] = 255;
					//img.at<Vec3b>(row, col)[0] = 0;
					//img.at<Vec3b>(row, col)[2] = 0;

					//Mat jubu = img(Rect(col - ((21 - 1) / 2), row - ((21 - 1) / 2), 21, 21));
					//namedWindow("����", 0);
					//cvResizeWindow("����", W, H);
					//imshow("����", jubu);
					//waitKey(1);

					//Ȩ�ؼ���
					Dsk = 0;
					err = 0;
					Dvk = 0;
					for (int wr = -((msS - 1) / 2); wr <= (msS - 1) / 2; wr++)//���ƶ�
					{
						for (int wc = -((msS - 1) / 2); wc <= (msS - 1) / 2; wc++)
						{
							try
							{
								Dsk += abs(imgcvt.at<uchar>(P[i].y + wr, P[i].x + wc) - imgcvt.at<uchar>(row + wr, col + wc));
							}
							catch (...)
							{
								err = 1;
								wr = msS;
								wc = msS;
							}
							
						}
					}
					for (int p = 0; p < k; p++)//�켣���ȳ̶�
					{
						Dvk += pow(P[i].intensityHistory[p]-intensityX0,2);
					}
					if (err == 0)
					{
						Dvk = Dvk/k;
						Dsk = Dsk / (msS*msS);
						Wk = exp(-((Dsk / (2 * SigmaG*SigmaG)) + (Dvk / (2 * SigmaV*SigmaV))));
						P[i].weight = Wk;
					}
					else
					{
						P[i].weight = 0;
					}
				}

				//��һ��Ȩ��
				normalize_weights(P, NUM_PARTICLE);

				//�ز�������
				//��Ȩ������
				qsort(P, NUM_PARTICLE, sizeof(particle), &particle_cmp);
				//����������ɾ
				/*for (int i = 0; i < NUM_PARTICLE; i++)
				{
					cout << P[i].weight*1000 << endl;
				}*/
				//��ʼ�ز���
				resample(P, NEW_P, NUM_PARTICLE);
				//qsort(P, NUM_PARTICLE, sizeof(particle), &particle_cmp);
				normalize_weights(P, NUM_PARTICLE);
			
				//waitKey(100);

			}
			
			//cvtColor(imgcvt, img, CV_GRAY2RGB);
			//for (int i = 0; i < NUM_PARTICLE; i++)
			//{
			//	img.at<Vec3b>(P[i].y, P[i].x)[1] = 0;
			//	img.at<Vec3b>(P[i].y, P[i].x)[0] = 0;
			//	img.at<Vec3b>(P[i].y, P[i].x)[2] = 255;
			//}
			//img.at<Vec3b>(row, col)[1] = 255;
			//img.at<Vec3b>(row, col)[0] = 0;
			//img.at<Vec3b>(row, col)[2] = 0;
			////Mat jubu = img(Rect(col - ((21 - 1) / 2), row - ((21 - 1) / 2), 21, 21));
			//namedWindow("����", 0);
			//cvResizeWindow("����", W, H);
			//imshow("����", img);
			//waitKey(0);
			//������
			intensityOutSum = 0;
			intensityOut = 0;
			for (int i = 0; i < NUM_PARTICLE; i++)
			{
				intensitySum = 0;
				intensityPartical = 0;
				for (int t = 0; t < T; t++)
				{
					intensitySum += P[i].intensityHistory[t];
				}
				intensityPartical = intensitySum / T;
				intensityOutSum += intensityPartical*P[i].weight;
			}
			intensityOut = MAX(0, MIN((int)(intensityOutSum), 255));
			imgOut.at<uchar>(row, col) = intensityOut;
		}
		//namedWindow("result", 1);
		//imshow("result", imgOut);
		//waitKey(10);
	}

	//����ͼ����ʾ
	namedWindow("noise image", 0);
	cvResizeWindow("noise image", W, H);
	imshow("noise image", imgcvt);
	//�����ʾ��������
	imshow("result", imgOut);

#ifdef FILE_OUT
	string output_path = OUTPUT_PATH;
	string output_file_name = output_path.append(OUTPUT_IMAGE_FILE_NAME);
	imwrite(output_file_name, imgOut);
#endif // FILE_OUT

	cout << "�ع����!" << endl;
	cv::waitKey(0);
	return 0;
}