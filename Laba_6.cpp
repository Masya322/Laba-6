#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "PolynomialRegression.h"
#include "opencv2/core/core.hpp"
#include <string>

using namespace std;
using namespace cv;



Mat frame;
Mat bord_frame;
Mat Chern_bel;

int main()
{
	float dst_size = 240.0;
	int wind_size = 40;
	Mat dst_bin_clone, window;
	Mat frame;
	Mat bord_frame;
	Mat Chern_bel;
	vector<float> left_coeff, right_coeff;

	VideoCapture cap("challenge.mp4");
				
	int vverh = 72;
	int vis_planki = 77;
	int niz = 88;
	int vis_planki2 = 91;
		
	namedWindow("Chern bel");
	int color01 = 0;
	int color02 = 0;
	int color03 = 32;
	int color11 = 233;
	int color12 = 255;
	int color13 = 255;
	

	while (1) {

		cap >> frame;

		if (frame.empty()) {			
			cap.set(CAP_PROP_POS_FRAMES, 0);
			continue;
		}

		float rows = frame.rows;
		float cols = frame.cols;

		float t1 = cols / 100.0*niz;
		float t2 = rows / 100.0*vis_planki2;
		float t3 = cols / 100.0*(100.0 - niz);

		float h1 = cols / 100.0*vverh;
		float h2 = rows / 100.0*vis_planki;
		float h3 = cols / 100.0*(100.0 - vverh);

		vector<Point2f> dst_t;


		vector<Point> polyline;
		polyline.push_back(Point(h1, h2));
		polyline.push_back(Point(h3, h2));
		polyline.push_back(Point(t3, t2));
		polyline.push_back(Point(t1, t2));


		vector<Point2f> points;
		points.push_back(Point2f(h1, h2));
		points.push_back(Point2f(h3, h2));
		points.push_back(Point2f(t3, t2));
		points.push_back(Point2f(t1, t2));


		vector<Point2f> dst_points;
		dst_points.push_back(Point2f(dst_size, 0.0));
		dst_points.push_back(Point2f(0.0, 0.0));
		dst_points.push_back(Point2f(0.0, dst_size));
		dst_points.push_back(Point2f(dst_size, dst_size));

		
		Mat _trap_ = frame.clone();
		Mat _trap_lin_ = _trap_.clone();
		polylines(_trap_, polyline, 1, Scalar(255, 0, 0), 4);
		
		Mat Matrix = getPerspectiveTransform(points, dst_points);
		Mat dst;
		warpPerspective(frame, dst, Matrix, Size(240, 240), INTER_LINEAR, BORDER_CONSTANT); 
				
		Mat dst_smena_cveta = dst.clone();
		cvtColor(dst_smena_cveta, dst_smena_cveta, COLOR_BGR2HLS);
		Mat dst_bin = dst_smena_cveta.clone();
		inRange(dst_bin, Scalar(color01, color02, color03), Scalar(color11, color12, color13), dst_bin);

		dst_bin_clone = dst_bin.clone();
		Chern_bel = dst_bin.clone(); 
		for (int i = 0; i < dst_size / wind_size; i++) {
			for (int j = 0; j < 2 * dst_size / wind_size - 1; j++) {
				Rect rect(j*wind_size / 2, i*wind_size, wind_size, wind_size);
				window = dst_bin(rect);
				Moments mom = moments(window, true);
				if (mom.m00 > 100) {
					Point2f point(j*wind_size / 2 + float(mom.m10 / mom.m00), i*wind_size + float(mom.m01 / mom.m00));

					bool dublicate = false;
					for (size_t n = 0; n < dst_t.size(); n++) {
						if (norm(dst_t[n] - point) < 10) {
							dublicate = true;
						}
					}
					if (!dublicate) {
						dst_t.push_back(point);
						circle(dst_bin_clone, point, 5, Scalar(128), -1);
					}
				}
			}
		}

		vector<Point2f> t_left;
		vector<Point2f> t_right;
		for (int i = 0; i < dst_t.size(); i++) {
			if (dst_t[i].x > dst_size / 2) {
				Point2f point(dst_t[i].x, dst_t[i].y);
				t_right.push_back(point);
			}
			else {
				Point2f point(dst_t[i].x, dst_t[i].y);
				t_left.push_back(point);
			}
		}

		if (t_left.size() > 3) {
			vector<float> coordinata_x;
			vector<float> coordinata_y;
			for (int i = 0; i < t_left.size(); i++) {
				coordinata_x.push_back(t_left[i].x);
				coordinata_y.push_back(t_left[i].y);
			}
			PolynomialRegression<float> left_polyn;
			left_polyn.fitIt(coordinata_y, coordinata_x, 2, left_coeff);
		}

		vector<Point2f> left_line;
		for (float i = 0.0; i < dst_bin_clone.rows; i += 0.1) {
			Point2f point(left_coeff[0] + left_coeff[1] * i + left_coeff[2] * pow(i, 2), i);
			left_line.push_back(point);
			circle(Chern_bel, point, 5, Scalar(128), -1);
		}

		if (t_right.size() > 3) {
			vector<float> coordinata_x;
			vector<float> coordinata_y;
			for (int i = 0; i < t_right.size(); i++) {
				coordinata_x.push_back(t_right[i].x);
				coordinata_y.push_back(t_right[i].y);
			}
			PolynomialRegression<float> right_polyn;
			right_polyn.fitIt(coordinata_y, coordinata_x, 2, right_coeff);
		}

		vector<Point2f> right_line;
		for (float i = 0.0; i < dst_bin_clone.rows; i += 0.1) {
			Point2f point(right_coeff[0] + right_coeff[1] * i + right_coeff[2] * pow(i, 2), i);
			right_line.push_back(point);
			circle(Chern_bel, point, 5, Scalar(128), -1);
		}

		
		if (left_line.size() > 0) {
			vector<Point2f> left_line_f;
			perspectiveTransform(left_line, left_line_f, Matrix.inv());
			for (int i = 0; i < left_line_f.size(); i++) {
				circle(_trap_lin_, left_line_f[i], 5, Scalar(255, 0, 0), -1);
			}
			float left_rad = abs(pow(1 + pow(left_coeff[1] + 2 * left_coeff[2] * dst_bin_clone.rows, 2), 3 / 2) / (2 * left_coeff[2]));
			
		}
				
		if (right_line.size() > 0) {
			vector<Point2f> right_line_f;
			perspectiveTransform(right_line, right_line_f, Matrix.inv());
			for (int i = 0; i < right_line_f.size(); i++) {
				circle(_trap_lin_, right_line_f[i], 5, Scalar(0, 0, 255), -1);
			}
			float right_rad = abs(pow(1 + pow(right_coeff[1] + 2 * right_coeff[2] * dst_bin_clone.rows, 2), 3 / 2) / (2 * right_coeff[2]));			
			
		}		
		
		

		//imshow("Frame", frame4poly);
		imshow("Rezultat", _trap_lin_);
		imshow("Vid sverhy", dst);		
		imshow("Chern bel s polosami", Chern_bel);
		imshow("Chern bel", dst_bin);
		//imshow("Dst bin Chern_bel", dst_bin_clone);


		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}

	cap.release();
	destroyAllWindows();

	return 0;
}


