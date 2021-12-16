#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <deque>


using namespace std;

//red
//cv::Scalar lowerColorBound = cv::Scalar(0, 40, 40); // in HSV
//cv::Scalar upperColorBound = cv::Scalar(20, 255, 255);

//green
cv::Scalar lowerColorBound = cv::Scalar(35, 40, 40); // in HSV
cv::Scalar upperColorBound = cv::Scalar(85, 255, 255);

cv::Scalar color_red = cv::Scalar(0, 0, 255);
cv::Scalar color_blue = cv::Scalar(255, 0, 0);
cv::Scalar color_black = cv::Scalar(0, 0, 0);


template <typename T, int MaxLen, typename Container = std::deque<T>>
class FixedQueue : public std::queue<T, Container> {
public:
	void push(const T& value) {
		if (this->size() == MaxLen) {
			this->c.pop_front();
		}
		std::queue<T, Container>::push(value);
	}
};

int frame_thick = 2;

int main()
{
	int dynamParams = 6;      // розмір вектора стану  [x,y, V_x,V_y, w,h]
	int measureParams = 4;   // розмір вектора спостереження (кількість вимірюваних параметрів)  [q_x, q_y, q_w, q_h]
	int controlParams = 0;
	cv::KalmanFilter Kalman(dynamParams, measureParams, controlParams, CV_32F);

	// error estimate covariance matrix 
	setIdentity(Kalman.errorCovPost, cv::Scalar::all(1));
	setIdentity(Kalman.errorCovPre, cv::Scalar::all(1));

	cv::Mat stateA(dynamParams, 1, CV_32F);
	cv::Mat measurementH(measureParams, 1, CV_32F);


	/* Transition State Matrix A == P == F
	[x, y, V_x, V_y, w, h]
	коваріаційна матриця стану, порядок матриці дорівнює розміру вектора стану
	ця матриця визначає "впевненість" фільтра в оцінці змінних станів. як виходить новий стан системи з попереднього
	Алгоритм самостійно оновлює цю матрицю у процесі роботи.
	*/
	cv::setIdentity(Kalman.transitionMatrix);


	/* Measure Matrix H == R
	[q_x, q_y, q_w, q_h]
	коваріаційна матриця шуму вимірювань, порядок матриці дорівнює розміру вектора спостереження
	Це значення, які отримують з датчиків системи.
	У багатьох випадках вважатимуться, що виміри не корелюють друг з одним.
	В цьому випадку матриця буде діагональною матрицею, де всі елементи поза головною діагоналі дорівнюють нулю.

	*/
	Kalman.measurementMatrix = cv::Mat::zeros(measureParams, dynamParams, CV_32F);
	Kalman.measurementMatrix.at<float>(0) = 1;
	Kalman.measurementMatrix.at<float>(7) = 1;
	Kalman.measurementMatrix.at<float>(16) = 1;
	Kalman.measurementMatrix.at<float>(23) = 1;


	/* Process Noise Covariance Matrix Q

	матриця помилки моделі, порядок матриці дорівнює розміру вектора стану
	коли фільтр передбачає стан системи, використовуючи модель процесу, він збільшує невпевненість в оцінці вектора стану
	*/
	setIdentity(Kalman.processNoiseCov, cv::Scalar::all(1e-1));
	Kalman.processNoiseCov.at<float>(14) = 5;
	Kalman.processNoiseCov.at<float>(21) = 5;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(Kalman.measurementNoiseCov, cv::Scalar::all(1e-1));

	cv::VideoCapture camera;
	if (!camera.open(0))
	{
		cout << "Camera is closed";
		return -1;
	}

	double ntime = 0;
	double ptime = 0;

	bool found = false;
	int searchTime = 0;

	const int size_q = 50;
	FixedQueue<float, size_q> px;
	FixedQueue<float, size_q> py;
	FixedQueue<float, size_q> mx;
	FixedQueue<float, size_q> my;

	cv::Mat frame;
	char exit = 'n';
	while (exit != 'y')
	{
		camera >> frame;

		int inRange_type = CV_8UC1;
		int blur_code = cv::ColorConversionCodes::COLOR_BGR2HSV;

		//  пошук фігур
		cv::Mat blured;
		blur(frame, blured, cv::Size(5, 5));

		cv::Mat inHsv;
		cvtColor(blured, inHsv, blur_code);

		cv::Mat inRange = cv::Mat::zeros(frame.size(), inRange_type);
		cv::inRange(inHsv, lowerColorBound, upperColorBound, inRange);

		int method = cv::CHAIN_APPROX_SIMPLE;
		int mode = cv::RetrievalModes::RETR_TREE;

		vector<vector<cv::Point>> contours;
		cv::findContours(inRange, contours, mode, method);


		vector<vector<cv::Point>> figures;
		vector<cv::Rect> redFrames;

		for (size_t i = 0; i < contours.size(); i++) {
			cv::Rect boundingRect;
			boundingRect = cv::boundingRect(contours[i]);

			if ((boundingRect.area() > 800) && (boundingRect.area() < 15000)) {
				figures.push_back(contours[i]);
				redFrames.push_back(boundingRect);
			}
		}

		cv::Mat frameCopy;
		frame.copyTo(frameCopy);
		for (size_t i = 0; i < figures.size(); i++)
		{
			// розташування яке ми "побачили" на відео
			cv::rectangle(frameCopy, redFrames[i], color_red, frame_thick);
		}


		// оновити predicted стан із measurement
		if (figures.size() > 0) {
			int w_half = redFrames[0].width / 2;
			int h_half = redFrames[0].height / 2;

			measurementH.at<float>(0) = redFrames[0].x + w_half;
			measurementH.at<float>(1) = redFrames[0].y + h_half;
			measurementH.at<float>(2) = (float)redFrames[0].width;
			measurementH.at<float>(3) = (float)redFrames[0].height;

			//зберегти
			mx.push(measurementH.at<float>(0));
			my.push(measurementH.at<float>(1));

			if (!found) {
				found = true;
				Kalman.errorCovPre, cv::Scalar::all(1);

				stateA.at<float>(0) = measurementH.at<float>(0);
				stateA.at<float>(1) = measurementH.at<float>(1);
				stateA.at<float>(2) = 0;     // швидкість невідома
				stateA.at<float>(3) = 0;
				stateA.at<float>(4) = measurementH.at<float>(2);
				stateA.at<float>(5) = measurementH.at<float>(3);

				Kalman.statePost = stateA;	//виправлений стан
			}
			else {
				Kalman.correct(measurementH);
			}
		}

		ptime = ntime;
		ntime = cv::getTickCount();
		double seconds = (ntime - ptime) / cv::getTickFrequency();
		if (found) {

			//  Matrix A        => є швидкість і час, отримуємо відстань
			Kalman.transitionMatrix.at<float>(2) = seconds;
			Kalman.transitionMatrix.at<float>(9) = seconds;

			stateA = Kalman.predict();


			cv::Rect blueFrame;
			blueFrame.width = stateA.at<float>(4);
			blueFrame.height = stateA.at<float>(5);

			int w_half = blueFrame.width / 2;
			int h_half = blueFrame.height / 2;

			// x,y - координати top-left кута прямокутника
			blueFrame.x = stateA.at<float>(0) - w_half;
			blueFrame.y = stateA.at<float>(1) - h_half;

			cv::rectangle(frameCopy, blueFrame, color_blue, frame_thick);      // розташування яке ми передбачаємо

			//зберегти
			px.push(stateA.at<float>(0));
			py.push(stateA.at<float>(1));
		}

		cv::imshow("Camera", frameCopy);
		exit = cv::waitKey(1);
	}

	int prev_px;
	int prev_py;
	int curr_px;
	int curr_py;

	int prev_mx;
	int prev_my;
	int curr_mx;
	int curr_my;

	int line_thickness = 1;
	int line_type = cv::LineTypes::LINE_8;
	int make_type = CV_8UC3;

	cv::Mat plot(frame.rows, frame.cols, make_type, color_black);
	if (!plot.data) {
		cout << "Could not open";
		return -1;
	}

	if ((!px.empty()) && (!mx.empty())) {
		prev_px = px.front();
		prev_py = py.front();
		px.pop();
		py.pop();

		prev_mx = mx.front();
		prev_my = my.front();
		mx.pop();
		my.pop();
	}
	while ((!px.empty()) && (!mx.empty())) {
		curr_px = px.front();
		curr_py = py.front();
		px.pop();
		py.pop();

		cv::Point ps(prev_px, prev_py);
		cv::Point pf(curr_px, curr_py);

		line(plot, ps, pf, color_blue, line_thickness, line_type);

		prev_px = curr_px;
		prev_py = curr_py;


		curr_mx = mx.front();
		curr_my = my.front();
		mx.pop();
		my.pop();

		cv::Point ms(prev_mx, prev_my);
		cv::Point mf(curr_mx, curr_my);


		line(plot, ms, mf, color_red, line_thickness, line_type);

		prev_mx = curr_mx;
		prev_my = curr_my;
	}


	imshow("Plot", plot);
	cv::waitKey(0);

	return 0;
}