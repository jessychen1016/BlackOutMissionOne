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
#include <opencv2/core/core.hpp>

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
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;
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
int stage_one_moving_count = 0;
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



 
    while (cvGetWindowHandle(window_name)){ // Application still alive?

		// make sure the ball is not in mouth
		ZActionModule::instance()->readData();
		if (ZActionModule::instance()->getInfrared()){
			for (int i = 0; i <=100; i++){
			ZActionModule::instance()->sendPacket(2, 0, 0, 0, true);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			cout << "888888888888888888888888888"<< endl;
			}
			break;
		}

		time0 = clock();
        auto start_time = clock();
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();
        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();
        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        // imshow ("image", color_mat);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);
        // imshow ("image_depth", depth_mat);
       
        Mat Gcolor_mat;
        Mat Gdepth_mat;


        GaussianBlur(color_mat,Gcolor_mat,Size(11,11),0);
        
        // Crop both color and depth frames
        Gcolor_mat = Gcolor_mat(crop);
        depth_mat = depth_mat(crop);

         //start of mod
        mat_rows = Gcolor_mat.rows;
        mat_columns =Gcolor_mat.cols;
        // cout<< mat_columns<< endl;
        Mat imgHSV;
        vector<Mat> hsvSplit;
        cvtColor(Gcolor_mat, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    
        //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
        split(imgHSV, hsvSplit);
        equalizeHist(hsvSplit[2],hsvSplit[2]);
        merge(hsvSplit,imgHSV);
        Mat imgThresholded;
    
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
    
        //开操作 (去除一些噪点)
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    
        //闭操作 (连接一些连通域)
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
		depth_m = depth_mat.at<double>((int)moment.m01 / moment.m00, (int)moment.m10 / moment.m00);
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
		length_to_mid = (moment.m10 / moment.m00 - 260)*depth_length_coefficient(magic_distance) / 320;
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
		if (turning_count <= 70){
			if (pixal_to_bottom != 480 ) {
				cout << "go to next state" << endl;
				break;
			}
			else {
				if (pixal_to_left == 0 && last_frame_length > 0) {
					ZActionModule::instance()->sendPacket(2, 0, 0, 120);
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					cout << "1111" << endl;
					turning_count++;
				}
				else if (pixal_to_left == 0 && last_frame_length < 0) {
					ZActionModule::instance()->sendPacket(2, 0, 0, -60);
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					cout << "2222" << endl;
					turning_count++;
				}
			}
		}
		else{
			if (pixal_to_bottom != 480 ) {
				cout << "go to next state" << endl;
				break;
			}
			else {
				for (stage_one_moving_count = 1; stage_one_moving_count <= 4; stage_one_moving_count ++){
					ZActionModule::instance()->sendPacket(2, stage_one_moving_count * 15, 0, 0, true);
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					cout << "3333" << endl;
				}
			}
			turning_count = 0;
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}


	}//end of while
	// second while for catching the stationary ball
	 while (cvGetWindowHandle(window_name)){ // Application still alive?

	//  make sure the ball is not in mouth
		ZActionModule::instance()->readData();
		if (ZActionModule::instance()->getInfrared()){
			for (int i = 0; i <=100; i++){
			ZActionModule::instance()->sendPacket(2, 0, 0, 0, true);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			cout << "999999999999999999999999999999"<< endl;
			}
			
			break;
		}
		time0 = clock();
        auto start_time = clock();
        // auto start_time_1 = clock();

        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        // imshow ("image", color_mat);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);
        // imshow ("image_depth", depth_mat);
    

        Mat Gcolor_mat;
        Mat Gdepth_mat;


        GaussianBlur(color_mat,Gcolor_mat,Size(11,11),0);
        
        

        // Crop both color and depth frames
        Gcolor_mat = Gcolor_mat(crop);
        depth_mat = depth_mat(crop);

         //start of mod
        mat_rows = Gcolor_mat.rows;
        mat_columns =Gcolor_mat.cols;
        // cout<< mat_columns<< endl;
        Mat imgHSV;
        vector<Mat> hsvSplit;
        cvtColor(Gcolor_mat, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    
        //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
        split(imgHSV, hsvSplit);
        equalizeHist(hsvSplit[2],hsvSplit[2]);
        merge(hsvSplit,imgHSV);
        Mat imgThresholded;
    
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
    
        //开操作 (去除一些噪点)
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    
        //闭操作 (连接一些连通域)
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
		depth_m = depth_mat.at<double>((int)moment.m01 / moment.m00, (int)moment.m10 / moment.m00);
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
		length_to_mid = (moment.m10 / moment.m00 - 260)*depth_length_coefficient(magic_distance) / 320;
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
				ZActionModule::instance()->sendPacket(2, 30, 0, 1.0 * length_to_mid + flag * 3);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				last_frame_length = length_to_mid;
				last_frame_pixal = pixal_to_bottom;
			}
		}
		//waitKey(3);
		cout << "All   " << 1000 * ((double)(clock() - time0)) / CLOCKS_PER_SEC << endl;

	}//end of while

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
