// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras
// Please see https://github.com/opencv/opencv/blob/master/LICENSE
// under the Intel® Core™ i5-3570 CPU @ 3.40GHz × 4 

#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include "cv-helpers.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <cmath>
#include <omp.h>
#include "iostream"
#include "time.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <actionmodule.h>



const size_t inWidth      = 600;
const size_t inHeight     = 900;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;
int mat_columns;
int mat_rows;
int length_to_mid;
int pixal_to_bottom;
int pixal_to_left;
double alpha=0;
using namespace cv;
using namespace cv::dnn;
using namespace rs2;
using namespace std;
using namespace rs400;

double depth_length_coefficient(double depth){
    
    double length;
    length = 48.033*depth+5.4556;
    return length;

}
    

int main(int argc, char** argv) try
{


    ZActionModule::instance();
    context ctx;
    auto devices = ctx.query_devices();
    size_t device_count = devices.size();
    if (!device_count)
    {
        cout <<"No device detected. Is it plugged in?\n";
        return EXIT_SUCCESS;
    }

    // Get the first connected device
    auto dev = devices[0];

    if (dev.is<rs400::advanced_mode>())
    {
        auto advanced_mode_dev = dev.as<rs400::advanced_mode>();
        // Check if advanced-mode is enabled
        if (!advanced_mode_dev.is_enabled())
        {
            // Enable advanced-mode
            advanced_mode_dev.toggle_advanced_mode(true);
        }
    }
    else
    {
        cout << "Current device doesn't support advanced-mode!\n";
        return EXIT_FAILURE;
    }
   
    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    std::ifstream t("/home/jessy/Desktop/rs_examples/config/labconfig4351.json");
    std::string str((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());
    rs400::advanced_mode dev4json = config.get_device();
    dev4json.load_json(str);
    

    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
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

    // double distance[60] = {0};
    double last_x_meter = 0;
    double this_x_meter = 0;
    double last_y_meter = 0;
    double this_y_meter = 0;
    double y_vel = 0;
    double x_vel = 0;
    double velocity;
    double alphaset[5] = {0};
    double alpha_mean=0;
    double move_distance=0;
    double first_magic_distance=5;
    int count = 0;
    int magic_distance_flag = 1;
    string move_direction ;
    int last_frame_length = 50;
    int last_frame_pixal = 480;
    // Declaring two concurrent queues that will be used to push and pop frames from different threads
    rs2::frame_queue depth_data;
    rs2::frame_queue color_data;
    // int para1 = 200 ,para2 = 50;

    // Atomic boolean to allow thread safe way to stop the thread
    std::atomic_bool stopped(false);

     // Create a thread for getting frames from the device and process them
    // to prevent UI thread from blocking due to long computations.
    std::thread processing_thread([&]() {
        while (!stopped) //While application is running
        {
            rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
            
            // Make sure the frames are spatially aligned
            data = align_to.process(data);

            auto color_frame = data.get_color_frame();
            auto depth_frame = data.get_depth_frame();
            if (!depth_frame) // Should not happen but if the pipeline is configured differently
                return;       //  it might not provide depth and we don't want to crash
            static int last_frame_number = 0;
            if (color_frame.get_frame_number() == last_frame_number) continue;
            last_frame_number = color_frame.get_frame_number();
            // Push filtered & original data to their respective queues
            // Note, pushing to two different queues might cause the application to display
            //  original and filtered pointclouds from different depth frames
            //  To make sure they are synchronized you need to push them together or add some
            //  synchronization mechanisms
            depth_data.enqueue(depth_frame);
            color_data.enqueue(color_frame);
        }
    });

    rs2::frame depth_frame;
    rs2::frame color_frame;
     while (cvGetWindowHandle(window_name))
    // for(int i = 0; i<60 && cvGetWindowHandle(window_name) ; i++)
    {
        auto start_time = clock();
        // Wait for the next set of frames
        // need to modify
        if (!depth_data.poll_for_frame(&depth_frame) || !color_data.poll_for_frame(&color_frame))
        continue;
       
        auto end_time_1 = clock();
        cout<<"time for recieving "<<1000.000*(end_time_1-start_time)/CLOCKS_PER_SEC<<std::endl;
        
        // If we only received new depth frame, 
        // but the color did not update, continue
        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        // imshow ("image", color_mat);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

        auto end_time_2 = clock();
        cout<<"time before Gauss "<<1000.000*(end_time_2-start_time)/CLOCKS_PER_SEC<<std::endl;
        // imshow ("image_depth", depth_mat);
        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
        
        
        Mat Gcolor_mat;
        // Mat Gdepth_mat;
        // auto end_time_1 = clock();
        // cout<<"time before Gaussian"<<1000.000*(end_time_1-start_time)/CLOCKS_PER_SEC<<std::endl;
        
        GaussianBlur(color_mat,Gcolor_mat,Size(11,11),0);
        //GaussianBlur(depth_mat,Gdepth_mat,Size(11,11),0);


        // Crop both color and depth frames
        Gcolor_mat = Gcolor_mat(crop);
        depth_mat = depth_mat(crop);


        auto end_time_3 = clock();
        cout<<"time before HSV "<<1000.000*(end_time_3-start_time)/CLOCKS_PER_SEC<<std::endl;
         //start of mod
        // mat_rows = Gcolor_mat.rows;
        // mat_columns =Gcolor_mat.cols;
        // cout<< mat_columns<< endl;
        // cout<< mat_rows<< endl;
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
        auto end_time_4 = clock();
        cout<<"time before contour "<<1000.000*(end_time_4-start_time)/CLOCKS_PER_SEC<<std::endl;
        // imshow("Thresholded Image", imgThresholded); //show the thresholded image
    //   imshow("Original", Gcolor_mat); //show the original image
    
        char key = (char) waitKey(1);
        //end of mod
    //    vector<Vec3f> circles;
    //    HoughCircles(imgThresholded,circles,CV_HOUGH_GRADIENT,1,1000,para1,para2,0,1000);
    //    Point hough_center(cvRound(circles[0][0]), cvRound(circles[0][1]));

        vector<vector<cv::Point>> contours ;
        cv::findContours(imgThresholded,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        double maxArea = 0;
        vector<cv::Point> maxContour;
        for(size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                maxContour = contours[i];
            }
        }
        cv::Rect maxRect = cv::boundingRect(maxContour);

        // auto object =  maxRect & Rect (0,0,depth_mat.cols, depth_mat.rows );
        auto object =  maxRect ;
        auto moment = cv::moments(maxContour,true);
        // std::cout<<"x="<< moment.m10 / moment.m00<<"  y="<<moment.m01/moment.m00<<std::endl;
        // std::cout<<"depth"<< depth_mat.at<double>((int)moment.m01 / moment.m00,(int)moment.m10/moment.m00)<<std::endl;
        
        // Calculate mean depth inside the detection region
        // This is a very naive way to estimate objects depth
        // but it is intended to demonstrate how one might 
        // use depht data in general
        Scalar depth_m;            
        if (moment.m00==0 ){
            moment.m00 = 1;
        }
        Point moment_center (moment.m10 / moment.m00, moment.m01/moment.m00);
        depth_m = depth_mat.at<double>((int)moment.m01 / moment.m00,(int)moment.m10/moment.m00);
        double magic_distance = depth_m[0] * 1.062;
        std::ostringstream ss;
        ss << " Ball Detected ";
        ss << std::setprecision(3) << magic_distance << " meters away" ;
        String conf(ss.str());
        // distance[i]=magic_distance;

        rectangle(Gcolor_mat, object, Scalar(0, 255, 0));
        int baseLine = 0;
        Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        auto center = (object.br() + object.tl())*0.5;
        center.x = center.x - labelSize.width / 2;
        center.y = center.y + 30 ;
        rectangle(Gcolor_mat, Rect(Point(center.x, center.y - labelSize.height),
        Size(labelSize.width, labelSize.height + baseLine)),
        Scalar(255, 255, 255), CV_FILLED);
        putText(Gcolor_mat, ss.str(), center,
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

        // velocity = sqrt(y_vel*y_vel + x_vel*x_vel);
        // // velocity window
        // ostringstream ss_v;
        // ss_v << " The speed is ";
        // ss_v <<setprecision(3) << velocity << " m/s" ;
        // String conf_v(ss_v.str());
        // Size labelSize_v = getTextSize(ss_v.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // auto center_v = (object.br() + object.tl())*0.5;
        // center_v.x = 160- labelSize_v.width / 2;
        // center_v.y = 475;

        // rectangle(Gcolor_mat, Rect(Point(center_v.x, center_v.y - labelSize_v.height),
        //     Size(labelSize_v.width, labelSize_v.height + baseLine)),
        //     Scalar(255, 255, 255), CV_FILLED);
        // putText(Gcolor_mat, ss_v.str(), center_v,
        //         FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

        // // move direction window
        
        // ostringstream ss_move_direction;
        // ss_move_direction << " Move to the ";
        // ss_move_direction << move_direction;
        // ss_move_direction << " for ";
        // ss_move_direction << setprecision(3) << move_distance << " cm" ;
        // String conf_move_direction(ss_v.str());
        // Size labelSize_move_direction = getTextSize(ss_move_direction.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // auto center_move_direction = (object.br() + object.tl())*0.5;
        // center_move_direction.x = 160- labelSize_move_direction.width / 2;
        // center_move_direction.y = 450;

        // rectangle(Gcolor_mat, Rect(Point(center_move_direction.x, center_move_direction.y - labelSize_move_direction.height),
        //     Size(labelSize_move_direction.width, labelSize_move_direction.height + baseLine)),
        //     Scalar(255, 255, 255), CV_FILLED);
        // putText(Gcolor_mat, ss_move_direction.str(), center_move_direction,
        //         FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
        



        // calculate length to midline
        length_to_mid = (moment.m10 / moment.m00-200)*depth_length_coefficient(magic_distance)/320;
        pixal_to_left = moment.m10 / moment.m00;
        pixal_to_bottom = (480-moment.m01 / moment.m00);
        // cout <<"pixal_to_bottom ="<<pixal_to_bottom<<"    ";
        cout << endl<<"length to midline ="<<length_to_mid<<"    ";
        if (magic_distance_flag ==1 && abs(length_to_mid) == 0){
            first_magic_distance = magic_distance;
            magic_distance_flag = 0;
        }
       
        
        imshow(window_name, Gcolor_mat);
        if (waitKey(1) >= 0) break;
        // imshow("heatmap", depth_mat);
        this_x_meter = magic_distance;
        this_y_meter = abs(length_to_mid);

        // x_vel = (this_x_meter - last_x_meter)/(end_time-start_time)*CLOCKS_PER_SEC;
        // std::cout<<1000.000*(end_time-start_time)/CLOCKS_PER_SEC<<std::endl;
        // cout<<"velocity = "<<x_vel<<"       ";
        // if(x_vel<-1){
        //     count += 1;
        //     alpha = atan(abs(last_y_meter - this_y_meter)/abs(this_x_meter-last_x_meter)/100);
        //     cout<<"alpha  ="<<alpha<<"      ";
        //     if( count <= 5){
        //     alphaset[count-1]=alpha;
        //     alpha_mean+= alphaset[count-1];
        //     }
        // }
        
        // if(count ==5 ){
        //     alpha_mean /= 5;
        //     cout<<"alpha mean=  "<<alpha_mean;
        //     move_distance = alpha_mean*first_magic_distance*100;
        //     cout<<endl<<"the depth for you to react ="<<first_magic_distance-magic_distance<<endl;
        //     count = 6;
        // }
        // if(length_to_mid < 0 ){
        //     move_direction = "left";
        // }
        // else{
        //     move_direction = "right";
        // }
        // last_x_meter = this_x_meter;
        // last_y_meter = this_y_meter;
        // cout<<"  first_magic_distance ="<<first_magic_distance<<endl;
        // cout<<"  move distance ="<<move_distance<<endl;

        // rotate the robot
        // cout<<"last pixal bottom    "<< last_frame_pixal<<endl;
        // cout<<"last frame length    "<< last_frame_length<<endl;
        // cout<<"pixal to left        "<< pixal_to_left<<endl;
        if (pixal_to_bottom == 480 && last_frame_pixal<100){
            ZActionModule::instance()->sendPacket(2, 10, 0, 0, true);
		    std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
            cout<<"0"<<endl;
        }
        else{
            if (pixal_to_left == 0 && last_frame_length > 0){
                ZActionModule::instance()->sendPacket(2, 0, 0, 30);
		        std::this_thread::sleep_for(std::chrono::milliseconds(1));  
                cout<<"1"<<endl;
            }
            else if(pixal_to_left == 0 && last_frame_length < 0){
                ZActionModule::instance()->sendPacket(2, 0, 0, -30);
		        std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
                cout<<"2"<<endl;
            }
            else{
                int flag = 1;
                if(length_to_mid >0){
                    flag = 1;
                }
                else if(length_to_mid < 0){
                    flag = -1;
                }
                else{
                    flag = 0;
                }
                ZActionModule::instance()->sendPacket(2, 50, 0, 1*length_to_mid+flag*5);
		        std::this_thread::sleep_for(std::chrono::milliseconds(1));                
                last_frame_length = length_to_mid;
                last_frame_pixal = pixal_to_bottom;
                cout<<"3"<<endl;
            }
        }
            auto end_time = clock();
            cout<<"time in a while"<<1000.000*(end_time-start_time)/CLOCKS_PER_SEC<<endl;
    }
    
    // double dmean = 0;
    // double dvariance = 0; 

    // for(int i=30 ; i<60; i++){

    //     dmean += distance[i];
    // }
    // dmean /= 30.0;

    // for (int i=30; i<60; i++ ){
    //     dvariance += (distance[i]-dmean) * (distance[i]-dmean);
    // }
    
    // dvariance /= 29;

    // std::cout << dmean << ",      "<< sqrt(dvariance) << std::endl;  
    stopped = true;
    processing_thread.join();
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

