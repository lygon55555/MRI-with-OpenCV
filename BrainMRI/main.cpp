#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void  calc_Histo(const Mat& image, Mat& hist, int bins, int range_max = 256)
{
	int		histSize[] = { bins };			// ������׷� ��ް���
	float   range[] = { 0, (float)range_max };		// ������׷� ����
	int		channels[] = { 0 };				// ä�� ���
	const float* ranges[] = { range };

	calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
}

void draw_histo(Mat hist, Mat &hist_img, Size size = Size(256, 200))
{
	hist_img = Mat(size, CV_8U, Scalar(255));
	float  bin = (float)hist_img.cols / hist.rows;
	normalize(hist, hist, 0, size.height, NORM_MINMAX);

	for (int i = 0; i < hist.rows; i++)
	{
		float idx1 = i * bin;
		float idx2 = (i + 1) * bin;
		Point2f pt1 = Point2f(idx1, 0);
		Point2f pt2 = Point2f(idx2, hist.at <float>(i));

		if (pt2.y > 0)
			rectangle(hist_img, pt1, pt2, Scalar(0), -1);
	}
	flip(hist_img, hist_img, 0);
}


void create_hist(Mat img, Mat &hist, Mat &hist_img) // ������׷� ���� �׷��� �׸��� ����
{
	int  histsize = 256, range = 256;
	calc_Histo(img, hist, histsize, range);			// ������׷� ���
	draw_histo(hist, hist_img);							// ������׷� �׷��� �׸���
}

void calc_direct(Mat Gy, Mat Gx, Mat& direct)
{
	direct.create(Gy.size(), CV_8U);

	for (int i = 0; i < direct.rows; i++){
		for (int j = 0; j < direct.cols; j++){
			float gx = Gx.at<float>(i, j);
			float gy = Gy.at<float>(i, j);
			int theat = int(fastAtan2(gy, gx) / 45);
			direct.at<uchar>(i, j) = theat % 4;
		}
	}
}

void supp_nonMax(Mat sobel, Mat  direct, Mat& dst)		// ���ִ밪 ����
{
	dst = Mat(sobel.size(), CV_32F, Scalar(0));

	for (int i = 1; i < sobel.rows - 1; i++) {
		for (int j = 1; j < sobel.cols - 1; j++) 
		{
			int   dir = direct.at<uchar>(i, j);				// ���� ��
			float v1, v2;
			if (dir == 0) {			// ���� ���� 0�� ����
				v1 = sobel.at<float>(i, j - 1);
				v2 = sobel.at<float>(i, j + 1);
			}
			else if (dir == 1) {		// ���� ���� 45��
				v1 = sobel.at<float>(i + 1, j + 1);
				v2 = sobel.at<float>(i - 1, j - 1);
			}
			else if (dir == 2) {		// ���� ���� 90��
				v1 = sobel.at<float>(i - 1, j);
				v2 = sobel.at<float>(i + 1, j);
			}
			else if (dir == 3) {		// ���� ���� 135��
				v1 = sobel.at<float>(i + 1, j - 1);
				v2 = sobel.at<float>(i - 1, j + 1);
			}

			float center = sobel.at<float>(i, j);
			dst.at<float>(i, j) = (center > v1 && center > v2) ? center : 0;
		}
	}
}

void trace(Mat max_so, Mat& pos_ck, Mat& hy_img, Point pt, int low)
{
	Rect rect(Point(0, 0), pos_ck.size());
	if (!rect.contains(pt)) return;			// ����ȭ���� ���� ���� Ȯ�� 

	if (pos_ck.at<uchar>(pt) == 0 && max_so.at<float>(pt) > low)
	{
		pos_ck.at<uchar>(pt) = 1;			// ���� �Ϸ� ��ǥ
		hy_img.at<uchar>(pt) = 255;			// ���� ����

		// ���� ��� �Լ�
		trace(max_so, pos_ck, hy_img, pt + Point(-1, -1), low);
		trace(max_so, pos_ck, hy_img, pt + Point( 0, -1), low);
		trace(max_so, pos_ck, hy_img, pt + Point(+1, -1), low);
		trace(max_so, pos_ck, hy_img, pt + Point(-1, 0), low);

		trace(max_so, pos_ck, hy_img, pt + Point(+1, 0), low);
		trace(max_so, pos_ck, hy_img, pt + Point(-1, +1), low);
		trace(max_so, pos_ck, hy_img, pt + Point( 0, +1), low);
		trace(max_so, pos_ck, hy_img, pt + Point(+1, +1), low);
	}
}

void  hysteresis_th(Mat max_so, Mat&  hy_img, int low, int high)
{
	Mat pos_ck(max_so.size(), CV_8U, Scalar(0));
	hy_img = Mat(max_so.size(), CV_8U, Scalar(0));

	for (int i = 0; i < max_so.rows; i++){
		for (int j = 0; j < max_so.cols; j++)
		{
			if (max_so.at<float>(i, j) > high)
				trace(max_so, pos_ck, hy_img, Point(j, i), low);
		}
	}
}

int main()
{
	//histogram equalize
	Mat image[4];
	Mat hist, dst1, dst2, hist_img, hist_img1, hist_img2;
	Mat gau_img, Gx, Gy, direct, sobel, max_sobel, hy_img, canny[4];

	// �� ���� �׷��̽����Ϸ� �б�
	image[0] = imread("C:/mri1.jpg", IMREAD_GRAYSCALE);	//����		
	image[1] = imread("C:/mri2.png", IMREAD_GRAYSCALE);	//������
	image[2] = imread("C:/mri3.jpg", IMREAD_GRAYSCALE);	//����
	image[3] = imread("C:/mri4.jpg", IMREAD_GRAYSCALE);	//������

	for (int k = 0; k < 4; k++)
	{
		CV_Assert(!image[k].empty());	//�������� ���� ó��

		create_hist(image[k], hist, hist_img);				// ������׷� �� �׷��� �׸���
																// ������׷� ������ ���
		Mat accum_hist = Mat(hist.size(), hist.type(), Scalar(0));
		accum_hist.at<float>(0) = hist.at<float>(0);
		for (int i = 1; i < hist.rows; i++) 
		{
			accum_hist.at<float>(i) = accum_hist.at<float>(i - 1) + hist.at<float>(i);
		}

		accum_hist /= sum(hist)[0];							// �������� ����ȭ
		accum_hist *= 255;
		dst1 = Mat(image[k].size(), CV_8U);
		for (int i = 0; i < image[k].rows; i++) {
			for (int j = 0; j < image[k].cols; j++) {
				int idx = image[k].at<uchar>(i, j);
				dst1.at<uchar>(i, j) = (uchar)accum_hist.at<float>(idx);
			}
		}

		equalizeHist(image[k], dst2);							// ������׷� ��Ȱȭ

		//gaussian + canny
		GaussianBlur(image[k], gau_img, Size(5, 5), 0.3);
		Sobel(gau_img, Gx, CV_32F, 1, 0, 3);
		Sobel(gau_img, Gy, CV_32F, 0, 1, 3);
		sobel = abs(Gx) + abs(Gy);

		calc_direct(Gy, Gx, direct);
		supp_nonMax(sobel, direct, max_sobel);
		hysteresis_th(max_sobel, canny[k], 100, 150);
	}

	for (int i = 0; i < 4; i+=2)
	{
		imshow("Normal", canny[i]);
		imshow("Disabled", canny[i+1]);
		resizeWindow("Normal", 300, 250);
		resizeWindow("Disabled", 300, 250);
		waitKey();
	}
	
	return 0;
}