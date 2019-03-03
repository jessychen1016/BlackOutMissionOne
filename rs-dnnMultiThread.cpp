#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include "example.hpp"
#include <imgui.h>
#include "imgui_impl_glfw.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <ctime>

#include <thread>
#include <chrono>
#include <mutex>
#include <Semaphore.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv-helpers.hpp"
#include "opencv2/dnn/dnn.hpp"

#include "actionmodule.h"

using namespace rs400;
using namespace rs2;
using namespace std;
using namespace cv;

Semaphore get_video_to_frame_transfer(0);
Semaphore frame_transfer_to_mean(0);

const size_t inWidth = 600;
const size_t inHeight = 900;
const float WHRatio = inWidth / (float)inHeight;
int mat_columns;
int mat_rows;
int length_to_mid;
int pixal_to_bottom;
int pixal_to_left;
double alpha = 0;
double last_x_meter = 0;
double this_x_meter = 0;
double last_y_meter = 0;
double this_y_meter = 0;
double y_vel = 0;
double x_vel = 0;
double velocity;
double alphaset[5] = { 0 };
double alpha_mean = 0;
double move_distance = 0;
double first_magic_distance = 5;
int turning_count = 0;
int magic_distance_flag = 1;
string move_direction;
int last_frame_length = 50;
int last_frame_pixal = 480;

std::chrono::milliseconds dura(100);
clock_t time0, time1, time2, end1, end2;
double depth_length_coefficient(double depth) {
	double length;
	length = 48.033*depth + 5.4556;
	return length;
}

mutex mtx1, mtx2;

template <typename T>
void change(T *p, T *q) {
	T tmp;
	tmp = *p;
	*p = *q;
	*q = tmp;
}

void get_video(rs2::pipeline &pipe, rs2::frameset *latest_frameset, rs2::frameset *new_frameset) {
	while (true) {
        time1 = clock();
		*new_frameset = pipe.wait_for_frames();
		mtx1.lock();
		cout << 11111111 << endl;
		change(latest_frameset, new_frameset);
		mtx1.unlock();
        end1 = clock();
        cout << "Thread1   " << 1000 * (end1 - time1)/ CLOCKS_PER_SEC << endl;
        get_video_to_frame_transfer.Signal();
	}
}

void frame_transfer(rs2::align &align_to, rs2::pipeline &pipe, rs2::frameset *latest_frameset, Mat *pt_color_mat, Mat *pt_new_color_mat, Mat *pt_depth_mat, Mat *pt_new_depth_mat) {
	while (true){
        time2 = clock();
        get_video_to_frame_transfer.Wait();
		mtx1.lock();
		cout << 222222222 << endl;
        auto processed = align_to.process(*latest_frameset);
		mtx1.unlock();
        rs2::video_frame color_frame = processed.get_color_frame();
        rs2::depth_frame depth_frame = processed.get_depth_frame();
        *pt_new_color_mat = frame_to_mat(color_frame);
        *pt_new_depth_mat = depth_frame_to_meters(pipe, depth_frame);
        std::chrono::milliseconds dura2(1000);
        // std::this_thread::sleep_for(dura2);
		mtx2.lock();
		cout << 3333333333 << endl;
        change(pt_color_mat, pt_new_color_mat);
        change(pt_depth_mat, pt_new_depth_mat);
		mtx2.unlock();
        frame_transfer_to_mean.Signal();
        end2 = clock();
        cout << "Thread2   " << 1000 * (end2 - time2) / CLOCKS_PER_SEC << endl;
    }
}

// void gaussianblur(Mat *pt_normal_color_mat, Mat *pt_G_color_mat, Mat *pt_G_latest_color_mat){
//     while (true){
//         GaussianBlur(*pt_normal_color_mat, *pt_G_color_mat, Size(11, 11), 0);
//         change(pt_G_color_mat, pt_G_latest_color_mat);
//     }
// }

int main(int argc, char * argv[]) try{
	context ctx;
	auto devices = ctx.query_devices();
	size_t device_count = devices.size();
	if (!device_count){
		cout << "No device detected. Is it plugged in?\n";
		return EXIT_SUCCESS;
	}
	auto dev = devices[0];
	if (dev.is<rs400::advanced_mode>()){
		auto advanced_mode_dev = dev.as<rs400::advanced_mode>();
		// Check if advanced-mode is enabled
		if (!advanced_mode_dev.is_enabled()){
			// Enable advanced-mode
			advanced_mode_dev.toggle_advanced_mode(true);
		}
	}
	else{
		cout << "Current device doesn't support advanced-mode!\n";
		return EXIT_FAILURE;
	}



    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    //Calling pipeline's start() without any additional parameters will start the first device
    // with its default streams.
    //The start function returns the pipeline profile which the pipeline used to start the device
    rs2::pipeline_profile profile = pipe.start();
    rs2::align align_to(RS2_STREAM_COLOR);
	////////////////////////
	auto config_profile = profile.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
	Size cropSize;
	if (config_profile.width() / (float)config_profile.height() > WHRatio){
		cropSize = Size(static_cast<int>(config_profile.height() * WHRatio),
			config_profile.height());
	}
	else{
		cropSize = Size(config_profile.width(),
			static_cast<int>(config_profile.width() / WHRatio));
	}

	Rect crop(Point((config_profile.width() - cropSize.width) / 2,
		(config_profile.height() - cropSize.height) / 2),
		cropSize);

	const auto window_name = "Display Image";
	namedWindow(window_name, WINDOW_AUTOSIZE);


	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 0;
	int iHighH = 38;
	int iLowS = 71;
	int iHighS = 255;
	int iLowV = 203;
	int iHighV = 255;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	std::ifstream config("/home/jessy/Desktop/rs_examples/config/labconfig4351.json");
	std::string str((std::istreambuf_iterator<char>(config)),
		std::istreambuf_iterator<char>());
	rs400::advanced_mode dev4json = profile.get_device();
	dev4json.load_json(str);

	/////////////////////
    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    // rs2_stream align_to = find_stream_to_align(profile.get_streams());

    // Create a rs2::align object.
    // rs2::align allows us to perform alignment of depth frames to others frames
    //The "align_to" is the stream type to which we plan to align depth frames.
    // rs2::align align(align_to);

    // Define a variable for controlling the distance to clip
    float depth_clipping_distance = 1.f;

    rs2::frameset new_frameset, latest_frameset;
	new_frameset = latest_frameset = pipe.wait_for_frames();

    std::thread t1 ([&]() {
        get_video(pipe, &latest_frameset, &new_frameset);
    });
    


	std::this_thread::sleep_for(dura);

	auto processed = align_to.process(latest_frameset);
	rs2::video_frame color_frame = processed.get_color_frame();
	rs2::depth_frame depth_frame = processed.get_depth_frame();
	Mat latest_color_mat = frame_to_mat(color_frame);
	Mat latest_depth_mat = depth_frame_to_meters(pipe, depth_frame);
	Mat new_color_mat = latest_color_mat;
	Mat new_depth_mat = latest_depth_mat;

	std::thread t2([&]() {
        frame_transfer(align_to, pipe, &latest_frameset, &latest_color_mat, &new_color_mat, &latest_depth_mat, &new_depth_mat);
     });
	
    // Mat new_Gcolor_mat = latest_color_mat;
    // Mat Gcolor_mat = latest_color_mat;
    // std::this_thread::sleep_for(dura);

    // std::thread t3([&]() {
    //     // time3 = clock();
    //     // cout<< "00000000"<<endl;
    //     gaussianblur(&latest_color_mat, &new_Gcolor_mat, &Gcolor_mat);
    //     // end3 = clock();
    //     // cout << "Thread3   " << 1000 * ((double)(end3 - time3)) / CLOCKS_PER_SEC << endl;
    //     // cout<< "55555555"<<endl;
    // });
    // t3.detach();
	std::this_thread::sleep_for(dura);
    // std::chrono::milliseconds dura2(1000);
    // std::this_thread::sleep_for(dura2);
	// first while to locate the stationary ball
    while (cvGetWindowHandle(window_name)){ // Application still alive?
		time0 = clock();
        frame_transfer_to_mean.Wait(); 
		mtx2.lock();
		Mat Gcolor_mat;
		GaussianBlur(latest_color_mat, Gcolor_mat, Size(11, 11), 0);
        // cvtColor(Gcolor_mat, Gcolor_mat, COLOR_RGB2BGR);
		// imshow(window_name, Gcolor_mat);
		Gcolor_mat = Gcolor_mat(crop);
		auto depth_mat2 = latest_depth_mat(crop);
		mtx2.unlock();
		Mat imgHSV;
		vector<Mat> hsvSplit;
		cvtColor(Gcolor_mat, imgHSV, COLOR_BGR2HSV);
		split(imgHSV, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, imgHSV);
		Mat imgThresholded;
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
		morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
		vector<vector<cv::Point>> contours;
		cv::findContours(imgThresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		double maxArea = 0;
		vector<cv::Point> maxContour;

		for (size_t i = 0; i < contours.size(); i++){
			double area = cv::contourArea(contours[i]);
			if (area > maxArea){
				maxArea = area;
				maxContour = contours[i];
			}
		}
		cv::Rect maxRect = cv::boundingRect(maxContour);

		// auto object =  maxRect & Rect (0,0,depth_mat.cols, depth_mat.rows );
		auto object = maxRect;
		auto moment = cv::moments(maxContour, true);

		Scalar depth_m;
		if (moment.m00 == 0) {
			moment.m00 = 1;
		}
		Point moment_center(moment.m10 / moment.m00, moment.m01 / moment.m00);
		depth_m = depth_mat2.at<double>((int)moment.m01 / moment.m00, (int)moment.m10 / moment.m00);
		double magic_distance = depth_m[0] * 1.062;
		std::ostringstream ss;
		ss << " Ball Detected ";
		ss << std::setprecision(3) << magic_distance << " meters away";
		String conf(ss.str());

		rectangle(Gcolor_mat, object, Scalar(0, 255, 0));
		int baseLine = 0;
		Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		auto center = (object.br() + object.tl())*0.5;
		center.x = center.x - labelSize.width / 2;
		center.y = center.y + 30;

		rectangle(Gcolor_mat, Rect(Point(center.x, center.y - labelSize.height),
			Size(labelSize.width, labelSize.height + baseLine)),
			Scalar(255, 255, 255), CV_FILLED);

		putText(Gcolor_mat, ss.str(), center,
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		/////////////////////////////////////////////////////////////////
		length_to_mid = (moment.m10 / moment.m00 - 220)*depth_length_coefficient(magic_distance) / 320;
		pixal_to_left = moment.m10 / moment.m00;
		pixal_to_bottom = (480 - moment.m01 / moment.m00);
		cout << endl << "length to midline =" << length_to_mid << "    ";
		if (magic_distance_flag == 1 && abs(length_to_mid) == 0) {
			first_magic_distance = magic_distance;
			magic_distance_flag = 0;
		}


		imshow(window_name, Gcolor_mat);
		if (waitKey(1) >= 0) break;
		// imshow("heatmap", depth_mat);
		this_x_meter = magic_distance;
		this_y_meter = abs(length_to_mid);
		cout << "pixal to bottom ="<< pixal_to_bottom << endl;
		if (turning_count < 10){
			if (pixal_to_bottom != 480 ) {
				break;
				cout << "go to next state" << endl;
			}
			else {
				if (pixal_to_left == 0 && last_frame_length > 0) {
					ZActionModule::instance()->sendPacket(2, 0, 0, 30);
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					cout << "1" << endl;
					turning_count++;
				}
				else if (pixal_to_left == 0 && last_frame_length < 0) {
					ZActionModule::instance()->sendPacket(2, 0, 0, -30);
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					cout << "2" << endl;
					turning_count++;
				}
			}
		}
		else{
			if (pixal_to_bottom != 480 ) {
				break;
				cout << "go to next state" << endl;
			}
			else {
				ZActionModule::instance()->sendPacket(2, 10, 0, 30);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			turning_count == 0;
		}


	}//end of while
	// second while for catching the stationary ball
	 while (cvGetWindowHandle(window_name)){ // Application still alive?
		time0 = clock();
        frame_transfer_to_mean.Wait(); 
		mtx2.lock();
		cout << 444444444 << endl;
		Mat Gcolor_mat;
		GaussianBlur(latest_color_mat, Gcolor_mat, Size(11, 11), 0);
        // cvtColor(Gcolor_mat, Gcolor_mat, COLOR_RGB2BGR);
		// imshow(window_name, Gcolor_mat);
		Gcolor_mat = Gcolor_mat(crop);
		auto depth_mat2 = latest_depth_mat(crop);
		mtx2.unlock();
		Mat imgHSV;
		vector<Mat> hsvSplit;
		cvtColor(Gcolor_mat, imgHSV, COLOR_BGR2HSV);
		split(imgHSV, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, imgHSV);
		Mat imgThresholded;
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
		morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
		vector<vector<cv::Point>> contours;
		cv::findContours(imgThresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		double maxArea = 0;
		vector<cv::Point> maxContour;

		for (size_t i = 0; i < contours.size(); i++){
			double area = cv::contourArea(contours[i]);
			if (area > maxArea){
				maxArea = area;
				maxContour = contours[i];
			}
		}
		cv::Rect maxRect = cv::boundingRect(maxContour);

		// auto object =  maxRect & Rect (0,0,depth_mat.cols, depth_mat.rows );
		auto object = maxRect;
		auto moment = cv::moments(maxContour, true);

		Scalar depth_m;
		if (moment.m00 == 0) {
			moment.m00 = 1;
		}
		Point moment_center(moment.m10 / moment.m00, moment.m01 / moment.m00);
		depth_m = depth_mat2.at<double>((int)moment.m01 / moment.m00, (int)moment.m10 / moment.m00);
		double magic_distance = depth_m[0] * 1.062;
		std::ostringstream ss;
		ss << " Ball Detected ";
		ss << std::setprecision(3) << magic_distance << " meters away";
		String conf(ss.str());

		rectangle(Gcolor_mat, object, Scalar(0, 255, 0));
		int baseLine = 0;
		Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		auto center = (object.br() + object.tl())*0.5;
		center.x = center.x - labelSize.width / 2;
		center.y = center.y + 30;

		rectangle(Gcolor_mat, Rect(Point(center.x, center.y - labelSize.height),
			Size(labelSize.width, labelSize.height + baseLine)),
			Scalar(255, 255, 255), CV_FILLED);

		putText(Gcolor_mat, ss.str(), center,
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		/////////////////////////////////////////////////////////////////
		length_to_mid = (moment.m10 / moment.m00 - 220)*depth_length_coefficient(magic_distance) / 320;
		pixal_to_left = moment.m10 / moment.m00;
		pixal_to_bottom = (480 - moment.m01 / moment.m00);
		cout << endl << "length to midline =" << length_to_mid << "    ";
		if (magic_distance_flag == 1 && abs(length_to_mid) == 0) {
			first_magic_distance = magic_distance;
			magic_distance_flag = 0;
		}


		imshow(window_name, Gcolor_mat);
		if (waitKey(1) >= 0) break;
		// imshow("heatmap", depth_mat);
		this_x_meter = magic_distance;
		this_y_meter = abs(length_to_mid);

		if (pixal_to_bottom == 480 && last_frame_pixal<100) {
			ZActionModule::instance()->sendPacket(2, 10, 0, 0, true);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			cout << "0" << endl;
		}
		else {
			if (pixal_to_left == 0 && last_frame_length > 0) {
				ZActionModule::instance()->sendPacket(2, 0, 0, 30);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				cout << "1" << endl;
			}
			else if (pixal_to_left == 0 && last_frame_length < 0) {
				ZActionModule::instance()->sendPacket(2, 0, 0, -30);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				cout << "2" << endl;
			}
			else {
				int flag = 1;
				if (length_to_mid >0) {
					flag = 1;
				}
				else if (length_to_mid < 0) {
					flag = -1;
				}
				else {
					flag = 0;
				}
				ZActionModule::instance()->sendPacket(2, 50, 0, 1.0 * length_to_mid + flag * 3);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				last_frame_length = length_to_mid;
				last_frame_pixal = pixal_to_bottom;
			}
		}
		//waitKey(3);
		cout << "All   " << 1000 * ((double)(clock() - time0)) / CLOCKS_PER_SEC << endl;

	}//end of while
    t1.detach();
    t2.detach();
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
