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

#define NEW_PDF //取消定义时直接使用相似的位置作为转移模型
#define ADD_NOISE //加噪声
//#define FILE_OUT //文件输出，路径在下方常量中设置



/************常量*************/
constexpr auto IMAGE_FILE_NAME = "fprint512.png"; //lena.tiff    timg.jpg    building.jpg     niose building.png  noise building16.png  barbara.png boat.png  fprint3.png
constexpr auto OUTPUT_PATH = "C:\\Users\\15876\\Desktop\\result\\fprint512";
constexpr auto OUTPUT_IMAGE_FILE_NAME = "\\fprint512_nosie16_result.png";
constexpr auto OUTPUT_PATH_Noiseimg = "C:\\Users\\15876\\Desktop\\result\\fprint512";
constexpr auto OUTPUT_NOISE_IMAGE_FILE_NAME = "\\fprint512_noise16.png";
constexpr auto NUM_PARTICLE = 60;//粒子数量
constexpr auto NUM_PDF = 10;
constexpr auto NUM_PDF2 = 10;
constexpr auto T = 6;//迭代次数
constexpr auto transS = 17;//转移模型尺度s
constexpr auto msS = 7;//计算方差和标准差的尺度
constexpr auto NUM_COL = 512;
constexpr auto NUM_ROW = 512;//图片大小，void meanstdvalue_img中用到，需和图片大小设置一样

static double meanMat[NUM_COL][NUM_ROW] = { 0 };
static double stdMat[NUM_COL][NUM_ROW] = { 0 };//储存每个点周围区域的均值和标准差


/*********粒子结构体************/
typedef struct particle {
	int x;			// 当前x坐标
	int y;			// 当前y坐标
	double mean;        //粒子周围局部区域均值
	double var;         //粒子周围局部区域方差
	double weight;		// 该粒子的权重
	int xHistory[T];
	int yHistory[T];   //位置记录
	double intensityHistory[T]; //所经过位置的所有深度
	//double weightHistory[T];//权重记录


} PARTICLE;


/*求均值和标准差*/
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

void meanstdvalue(Mat scr, int col, int row, int windowsize, double* mean, double* std)//输入整个图，行数列数，窗口大小，返回值的地址
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


/*************************粒子初始化******************************************/
void particle_init(particle* particles, int x, int y)
{
	for (int i = 0; i < NUM_PARTICLE; i++)
	{
		//所有粒子初始化到所求粒子位置
		particles[i].x = x;
		particles[i].y = y;
		//particles[i].xPre = particles[i].x;
		//particles[i].yPre = particles[i].y;
		particles[i].xHistory[T] = { 0 };
		particles[i].yHistory[T] = { 0 };
		particles[i].intensityHistory[T] = { 0 };

		//权重全部为0
		particles[i].weight = 0;
	}
}


//挑选像素领域内均值方差相似的点,for的ij反了
int selectpixels(Mat imgcvt, int col, int row, int windowsize, double meanthv, double stdthv, int d[NUM_PARTICLE][2])
{
	double stdX0 = 0;
	double meanX0 = 0;
	double stdx = 0;
	double meanx = 0;
	int numVector = 0;
	//排序使用的变量
	double meanstd[NUM_PARTICLE];
	for (int i = 0; i < NUM_PARTICLE; i++)
	{
		meanstd[i] = 10000;
	}
	double meanstdx;

	//// 可删
	//Mat img;
	//cvtColor(imgcvt, img, CV_GRAY2RGB);

	//windowsize = 41;
	//meanthv = 3;
	//stdthv = 1.5;
	//meanstdvalue(imgcvt, col, row, 7, &stdX0, &meanX0);//效率低去掉，换成表格
	stdX0 = stdMat[col][row];
	meanX0 = meanMat[col][row];
	
	for (int i = col - (windowsize - 1) / 2; i <= col + (windowsize - 1) / 2; i++)
	{
		for (int j = row - (windowsize - 1) / 2; j <= row + (windowsize - 1) / 2; j++)
		{
			//meanstdvalue(imgcvt, i, j, 7, &stdx, &meanx);//效率低去掉，换成表格
			stdx = stdMat[i][j];
			meanx = meanMat[i][j];
			if (((-meanthv) < (meanx - meanX0)) && ((meanx - meanX0) < meanthv) && ((-stdthv) < (stdx - stdX0)) && ((stdx - stdX0) < stdthv))
			{
				numVector++;
				//选出最大的Np个，当做转移模型,排序使用均值加标准差分别*阈值的倒数
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

				//显示部分可以删了
				//相似区域涂红
				
				//img.at<Vec3b>(j, i)[1] = 0;
				//img.at<Vec3b>(j, i)[0] = 0;
				//img.at<Vec3b>(j, i)[2] = 255;
			}

		}
	}
	/*if (numVector < 60)
	{

	}*/
	////显示部分,可以删了
	////标出中心点，绿色
	//img.at<Vec3b>(row, col)[1] = 255;
	//img.at<Vec3b>(row, col)[0] = 0;
	//img.at<Vec3b>(row, col)[2] = 0;
	////显示相似像素点
	//Mat jubu = img(Rect(col - ((windowsize - 1) / 2), row - ((windowsize - 1) / 2), 40, 40));
	//namedWindow("jubu", 0);
	//cvResizeWindow("jubu", 512, 512);
	//imshow("jubu", img);
	//waitKey(0);

	return MIN(60, numVector);;
}

//k近邻法概率密度估计
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
	//计算区域内每个点到每个样本的距离，并排序方便寻找K个最近的样本点
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
	//K近邻，计算每个像素的概率k/n/v
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
	//归一化
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


/***********粒子状态转移**********/
//相关定义
/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 0.05
#define TRANS_Y_STD 0.05
/* autoregressive dynamics parameters for transition model */
#define A1  2.0//2.0
#define A2  -1.0//-1.0
#define B0  10
//以相似像素位移d为转移模型时的转移函数，NUM_PDF不同，正式程序中不使用
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

//以相似像素位移d为转移模型时的转移函数，正式程序中不使用
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

//根据概率密度函数，随机采样
particle transition_new(particle p, gsl_rng* rng, double transmodel[transS][transS], int H, int W)//参数说明：粒子，随机数种子，转移模型，图像宽，图像高
{
	int x, y;
	double u;
	double L_model[transS*transS] = { 0 };
	double cdf[transS*transS] = { 0 };
	particle pn = p;

	//均匀随机数
	u = gsl_ran_flat(rng, 0, 1);
	//折成一维的，一行一行，（其实可以不用拆，这里方便处理，牺牲了效率）
	for (int i = 0; i < transS; i++)
	{
		for (int j = 0; j < transS; j++)
		{
			L_model[i*transS + j] = transmodel[j][i];
		}
	}
	//状态转移模型的cdf
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


//权重归一化
void normalize_weights(particle* particles, int n)//参数说明：粒子地址，粒子数量
{
	float sum = 0;
	int i;

	for (i = 0; i < n; i++)
		sum += particles[i].weight;
	for (i = 0; i < n; i++)
		particles[i].weight /= sum;
}

//配合qsort，粒子按权重排序
int particle_cmp(const void* p1, const void* p2)
{
	//这个函数配合qsort，如果这个函数返回值: (1) <0时：p1排在p2前面   (2)  >0时：p1排在p2后面
	particle* _p1 = (particle*)p1;
	particle* _p2 = (particle*)p2;
	//这里就由大到小排序了
	return (int)((_p2->weight - _p1->weight)*1000000);//建议改三目运算符（还没改）
}
 
//二分法求数组中大于给定值的最小值索引
int get_min_index(double *array, int length, double _value)
{
	int _index = (length - 1) / 2;
	int last_index = length - 1;
	int _index_up_limit = length - 1;
	int _index_down_limit = 0;
	//先判断极值
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

//重采样
void resample(particle* particles, particle* new_particles, int num_particles)
{
	//计算每个粒子的概率累计和
	double sum[NUM_PARTICLE], temp_sum = 0;
	int k = 0;
	for (int j = num_particles - 1; j >= 0; j--) {
		temp_sum += particles[j].weight;
		sum[j] = temp_sum;
	}
	//为每个粒子生成一个均匀分布【0，1】的随机数
	RNG sum_rng(time(NULL));
	double Ran[NUM_PARTICLE];
	for (int j = 0; j < num_particles; j++) {
		sum_rng = sum_rng.next();
		Ran[j] = sum_rng.uniform(0.0, 1.0);
	}
	//在粒子概率累积和数组中找到最小的大于给定随机数的索引，复制该索引的粒子一次到新的粒子数组中 【从权重高的粒子开始】
	for (int j = 0; j < num_particles; j++) {
		int copy_index = get_min_index(sum, num_particles, Ran[j]);
		new_particles[k++] = particles[copy_index];
		if (k == num_particles)
			break;
	}
	//如果上面的操作完成，新粒子数组的数量仍少于原给定粒子数量，则复制权重最高的粒子，直到粒子数相等
	while (k < num_particles)
	{
		new_particles[k++] = particles[0]; //复制权值最高的粒子
	}
	//以新粒子数组覆盖久的粒子数组
	for (int i = 0; i < num_particles; i++)
	{
		particles[i] = new_particles[i];  //复制新粒子到particles
	}
}

double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
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
	//判断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//添加高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] + generateGaussianNoise(0, 0.5) * 16;//原来是*32
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
	namedWindow("原始图像", 2);
	cvResizeWindow("原始图像", imgcvt.cols, imgcvt.rows);
	imshow("原始图像", imgcvt);
	waitKey(100);

	//加噪声并保存噪声图片
#ifdef ADD_NOISE
	imgcvt = addGaussianNoise(imgcvt);
#ifdef FILE_OUT
	imgcvt = addGaussianNoise(imgcvt);
	string output_path_noiseimg = OUTPUT_PATH_Noiseimg;
	string output_file_name_noiseimg = output_path_noiseimg.append(OUTPUT_NOISE_IMAGE_FILE_NAME);
	imwrite(output_file_name_noiseimg, imgcvt);
#endif //FILE_OUT
#endif // ADD_NOISE

	PARTICLE P[NUM_PARTICLE];//粒子
	PARTICLE NEW_P[NUM_PARTICLE];

	//GSL随机数初始化
	gsl_rng* rng;
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, time(NULL));

	////显示粒子时使用
	//Mat img;
	//cvtColor(imgcvt, img, CV_GRAY2RGB);

	//输出的图像
	Mat imgOut;
	imgOut = imgcvt.clone();

	//局部均值和方差，避免重复计算，存入内存
	int intensityX0;
	double  stdX0, meanX0;
	double stdx, meanx;
	int d[NUM_PARTICLE][2] = { 0 };
	static int W = imgcvt.cols;
	static int H = imgcvt.rows;
	//权重计算变量
	double Dsk = 0;
	int SigmaG = 15;
	double Dvk = 0;
	int SigmaV = 15;
	double Wk = 0;
	int err = 0;
	//重构值计算变量
	double intensitySum = 0;
	double intensityPartical = 0;
	double intensityOutSum = 0;
	int intensityOut = 0;
	//转移模型
	double transmodel[transS][transS];
	int num_d = 0;

	cout << "图片的分辨率为： " << W << "*" << H << endl;
	cout << "设置的分辨率为：" << NUM_COL << "*" << NUM_ROW <<"  ";
	if (NUM_COL == W && NUM_ROW == H)
		cout << endl;
	else
	{
		cout << "设置的分辨率错误！！！修改NUM_COL、NUM_ROW的值！" << endl;
		cout << "重构未进行!"<< endl;
		waitKey(0);
		return -1;
	}

	//计算整个图中所有像素局部的方差和均值，存在meanmat和stdmat里
	meanstdvalue_img(imgcvt, msS);
	
	//下面两个for循环，遍历整个图像
	for (int row = 30; row < (H - 30); row++)
	{
		cout << "第" << row << "行; " << endl;
		for (int col = 30; col < (W - 30); col++)
		{
			//cout << "第" << col << "列; " ;
			for (int k = 0; k < T; k++)
			{
				//每个像素独立处理
				intensityX0 = imgcvt.at<uchar>(row, col);
				if(k==0)
				{
					//初始化位移d（转移模型），计算方差和均值相似度前<粒子数量>个位移d
					for (int i = 0; i < NUM_PARTICLE; i++)
					{
						d[i][0] = 0;
						d[i][1] = 0;
					}
					//计算转移模型
					num_d = selectpixels(imgcvt, col, row, transS, 5, 3, d);//3,1.5
#ifdef NEW_PDF
					//（新）转移模型
					transition_modeling(2, d, num_d, transmodel);
#endif // NEW_PDF

					//初始化
					particle_init(P, col, row);
				}
				//对每个粒子的操作
				////显示，可删,显示每个粒子
				//cvtColor(imgcvt, img, CV_GRAY2RGB);
				for (int i = 0; i < NUM_PARTICLE; i++)
				{
					//采样
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
					////显示，可删，显示每个粒子
					//img.at<Vec3b>(P[i].y, P[i].x)[1] = 0;
					//img.at<Vec3b>(P[i].y, P[i].x)[0] = 0;
					//img.at<Vec3b>(P[i].y, P[i].x)[2] = 255;

					//img.at<Vec3b>(row, col)[1] = 255;
					//img.at<Vec3b>(row, col)[0] = 0;
					//img.at<Vec3b>(row, col)[2] = 0;

					//Mat jubu = img(Rect(col - ((21 - 1) / 2), row - ((21 - 1) / 2), 21, 21));
					//namedWindow("粒子", 0);
					//cvResizeWindow("粒子", W, H);
					//imshow("粒子", jubu);
					//waitKey(1);

					//权重计算
					Dsk = 0;
					err = 0;
					Dvk = 0;
					for (int wr = -((msS - 1) / 2); wr <= (msS - 1) / 2; wr++)//相似度
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
					for (int p = 0; p < k; p++)//轨迹均匀程度
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

				//归一化权重
				normalize_weights(P, NUM_PARTICLE);

				//重采样部分
				//按权重排序
				qsort(P, NUM_PARTICLE, sizeof(particle), &particle_cmp);
				//排序结果，可删
				/*for (int i = 0; i < NUM_PARTICLE; i++)
				{
					cout << P[i].weight*1000 << endl;
				}*/
				//开始重采样
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
			//namedWindow("粒子", 0);
			//cvResizeWindow("粒子", W, H);
			//imshow("粒子", img);
			//waitKey(0);
			//计算结果
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

	//噪声图像显示
	namedWindow("noise image", 0);
	cvResizeWindow("noise image", W, H);
	imshow("noise image", imgcvt);
	//结果显示并保存结果
	imshow("result", imgOut);

#ifdef FILE_OUT
	string output_path = OUTPUT_PATH;
	string output_file_name = output_path.append(OUTPUT_IMAGE_FILE_NAME);
	imwrite(output_file_name, imgOut);
#endif // FILE_OUT

	cout << "重构完成!" << endl;
	cv::waitKey(0);
	return 0;
}