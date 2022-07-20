#include <fstream>
#include <sstream>
#include <iostream>

#include<algorithm>
#include <string>
#include <vector>
#include <queue>
#include <map>

#include <mutex>
#include <condition_variable>

#include <thread>
// #include <boost/thread/thread.hpp>
#include <termio.h>
#include<chrono>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/String.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;

int pointcloud_mode;
std::string data_dir;
double start_time;
double end_time;

ifstream data_file;
std::string data_str;
std::string data_file_type;

ifstream depth_list_file;
std::string depth_str;
double depth_time;


ros::Publisher pub_pointcloud;
ros::Publisher pub_path;
ros::Publisher pub_odom;

nav_msgs::Path path_msg;
nav_msgs::Odometry odom_msg;
sensor_msgs::PointCloud2 point_cloud_msg;
std_msgs::Header point_cloud_header;
std_msgs::Header camera_odom_header;
std_msgs::Header img_header;

bool begin_pub = false;
uint64_t time0 = 0;
int mark_id = -1;
tf::Transform point_cloud_tf;

double camera_factor;
double camera_cx;
double camera_cy;
double camera_fx;
double camera_fy;

Eigen::Matrix3d camera_R;
Eigen::Vector3d camera_t;
double camera_time;

void generatePointCloud(cv::Mat& rgb_img,cv::Mat &depth_img){
    // point_cloud_msg.header.stamp = ros::Time::now();
    // point_cloud_msg.header.frame_id = "base_link"; 
    // point_cloud_msg.header.frame_id = "map"; 
    point_cloud_msg.height = rgb_img.rows; 
    point_cloud_msg.width = rgb_img.cols;
    point_cloud_msg.is_bigendian = false;
    point_cloud_msg.point_step = 32;
    point_cloud_msg.row_step = 20480;
    point_cloud_msg.is_dense = true;

    // 新建一个点云
    pcl::PointCloud<pcl::PointXYZRGB> pointCloud;
    for (int v = 0; v < rgb_img.rows; v++){
        for (int u = 0; u < rgb_img.cols; u++) {
            int rr = rgb_img.at<cv::Vec3b>(v,u)[0];
            int gg = rgb_img.at<cv::Vec3b>(v,u)[1];
            int bb = rgb_img.at<cv::Vec3b>(v,u)[2];
            int dd = depth_img.at<ushort>(v,u); // 深度值

            if (dd < 0.01) continue; // 为0表示没有测量到
            double zz = 1.0 * dd / camera_factor;
            double xx = (u - camera_cx) / camera_fx * zz;
            double yy = (v - camera_cy) / camera_fy * zz;
            
            Eigen::Vector3d vec_point;
            vec_point[0] = xx;
            vec_point[1] = yy;
            vec_point[2] = zz;

            if(pointcloud_mode == 2){
                vec_point = camera_R * vec_point + camera_t;
            }

            pcl::PointXYZRGB p;
            p.x = vec_point[0];
            p.y = vec_point[1];
            p.z = vec_point[2];
            p.b = bb;
            p.g = gg;
            p.r = rr;
            pointCloud.points.push_back(p);
            // std::cout<<vec_point[0]<<" ";
        }
    }
    // pcl::toROSMsg (pcl::PointCloud<pcl::PointXYZRGB>,sensor_msgs::PointCloud2);
    pcl::toROSMsg(pointCloud,point_cloud_msg);
    // point_cloud_header.stamp = ros::Time::now();
    point_cloud_header.stamp = img_header.stamp;
    if(pointcloud_mode == 2){
        point_cloud_header.frame_id = "world";
    }else {
        point_cloud_header.frame_id = "camera";
    }

    point_cloud_msg.header = point_cloud_header;
    // point_cloud_msg.header.frame_id = "world"; 
    return;
}

void generateOdomPath(const Eigen::Matrix3d &RR,const Eigen::Vector3d &tt){
    geometry_msgs::PoseStamped pose_msg_tmp;
    pose_msg_tmp.header.frame_id = "world";
    pose_msg_tmp.pose.position.x = tt.x();
    pose_msg_tmp.pose.position.y = tt.y();
    pose_msg_tmp.pose.position.z = tt.z();
    Eigen::Quaterniond qq = Eigen::Quaterniond(RR);
    pose_msg_tmp.pose.orientation.x = qq.x();
    pose_msg_tmp.pose.orientation.y = qq.y();
    pose_msg_tmp.pose.orientation.z = qq.z();
    pose_msg_tmp.pose.orientation.w = qq.w();

    path_msg.poses.push_back(pose_msg_tmp);
    odom_msg.pose.pose = pose_msg_tmp.pose;
}

void rgbd_callback(cv::Mat &rgb_img, cv::Mat &depth_img){
    generatePointCloud(rgb_img,depth_img);
    begin_pub = true;
    cv::imshow("rgb_img",rgb_img);
    cv::imshow("depth_img",depth_img);
    cv::waitKey(10);
}

void camera_pose_callback(const Eigen::Matrix3d &RR,const Eigen::Vector3d &tt){
    camera_R = RR;
    camera_t = tt; 
    // camera_time = 0;
    generateOdomPath(RR,tt);
}

void camera_tf_callback(const Eigen::Matrix3d &RR,const Eigen::Vector3d &tt){
    Eigen::Quaterniond qq = Eigen::Quaterniond(RR);

// Quaternion(const tfScalar& x, const tfScalar& y, const tfScalar& z, const tfScalar& w) 
    tf::Quaternion tf_camera_rotation(qq.x(),qq.y(),qq.z(),qq.w());
    tf::Vector3 tf_camera_translation(tt.x(),tt.y(),tt.z());
    point_cloud_tf = tf::Transform (tf_camera_rotation, tf_camera_translation);

    generateOdomPath(RR,tt);
}

void timer_callback(const ros::TimerEvent& e)
{
    if(!begin_pub){
        return;
    }

    ROS_INFO("pointcloud publish");
    pub_pointcloud.publish(point_cloud_msg);
    pub_path.publish(path_msg);
    pub_odom.publish(odom_msg);
}

void timer_callback2(const ros::TimerEvent& e)
{
    if(pointcloud_mode == 2){
        return;
    }

    if(!begin_pub){
        return;
    }
    ROS_INFO("tf publish");
    static tf::TransformBroadcaster tf_broadcaster;
    std::cout<<"header= "<<point_cloud_msg.header.frame_id<<std::endl;
    tf_broadcaster.sendTransform(tf::StampedTransform(point_cloud_tf, point_cloud_header.stamp, "world", point_cloud_header.frame_id));
}

void timer_callback0(const ros::TimerEvent& e)
{
    std::cout<<"timer_callback0"<<std::endl;
    getline(data_file, data_str);
    std::cout<<"Data "<<data_str<<std::endl;

    stringstream iss_data(data_str);
     double img_time,tx,ty,tz,qw,qx,qy,qz;
    iss_data>>img_time>>tx>>ty>>tz>>qw>>qx>>qy>>qz;

    std::cout<<img_time<<"\t"<<depth_time<<std::endl;

    if(depth_time - img_time>=0.01){
        return;
    }else if(img_time - depth_time>=0.01){
        while(!depth_list_file.eof()){
            getline(depth_list_file, depth_str);
            stringstream iss_data(depth_str);
            iss_data>>depth_time;
            std::cout<<depth_time<<" "<<abs(depth_time - img_time)<<std::endl;
            if(abs(depth_time - img_time)<0.01){
                break;
            }
        }
    }

    std::string img_name = std::to_string(img_time);
    std::string depth_name = std::to_string(depth_time);

    cv::Mat  rgb_img = cv::imread(data_dir + "/rgb/" +img_name +".png");
    cv::Mat  depth_img = cv::imread(data_dir + "/depth/" +depth_name +".png");

if(1){
    std::cout<<data_dir + "/rgb/" +img_name +".png"<<std::endl;
    std::cout<<data_dir + "/depth/" +depth_name +".png"<<std::endl;
    std::cout<<std::endl;
}

    Eigen::Matrix3d camera_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d camera_t = Eigen::Vector3d::Zero();

    if(pointcloud_mode ==1){
         Eigen::Quaterniond camera_q(qw,qx,qy,qz);
         camera_R = Eigen::Matrix3d(camera_q);
         camera_t =Eigen::Vector3d(tx,ty,tz);
    }
   
    if(pointcloud_mode ==2){
        camera_pose_callback(camera_R,camera_t);
    }else{
        camera_tf_callback(camera_R,camera_t);
    }

    rgbd_callback(rgb_img,depth_img);
    begin_pub = true; 
}

int main(int argc, char **argv)
{
    camera_factor = 1000.0;
    camera_cx = 319.16667152289227;
    camera_cy = 235.58360480225772;
    camera_fx = 609.70550296798035;
    camera_fy = 609.09579671294716;

    pointcloud_mode = 0; //pointclond on the origin point
    // pointcloud_mode = 1; //pointcloud by tf
    // pointcloud_mode = 2; //point clood by pose

    ros::init(argc, argv, "ego_estimator");
    ros::NodeHandle nh("~");

    nh.param("/rgbd2pointcloud/pointcloud_mode", pointcloud_mode, pointcloud_mode);
    nh.param("/rgbd2pointcloud/data_dir", data_dir, data_dir);
    nh.param("/rgbd2pointcloud/start_time", start_time, start_time);
    nh.param("/rgbd2pointcloud/end_time", end_time, end_time);

    std::cout<<"pointcloud_mode "<<pointcloud_mode<<std::endl;
    std::cout<<"data_dir "<<data_dir<<std::endl;
    std::cout<<"start_time "<<start_time<<std::endl;
    std::cout<<"end_time "<<end_time<<std::endl;

    std::string data_file_name = data_dir + "/rgbd_pose.txt";
    std::cout<<data_file_name<<std::endl;
    data_file_type = "rgbd_pose" ;

	data_file.open(data_file_name);
    getline(data_file, data_str);

    double last_cam_time;
    while(!data_file.eof()){
        getline(data_file, data_str);
        stringstream iss_data(data_str);
        double time_tmp;
        iss_data>>time_tmp;
        if(time_tmp>=start_time){
            last_cam_time = time_tmp;
			std::cout<<"start time "<<start_time<<" data has found!"<<std::endl;
			break;
		}
    }

    std::string depth_list_file_name = data_dir + "/realsense_depth.txt";
	depth_list_file.open(depth_list_file_name);
    depth_time = 0;

    while(!depth_list_file.eof()){
        getline(depth_list_file, depth_str);
        stringstream iss_data(depth_str);
        iss_data>>depth_time;
        if(depth_time>start_time - 0.05){
			std::cout<<"start time "<<start_time<<" data has found!"<<std::endl;
			break;
		}
    }

    path_msg.header.frame_id = "world";
    odom_msg.header.frame_id = "world";

    pub_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/visualization_point_cloud", 100);
    pub_path = nh.advertise<nav_msgs::Path>("/visualization_path", 100);
    pub_odom = nh.advertise<nav_msgs::Odometry>("/visualization_odom", 100);

    ros::Timer timer1 = nh.createTimer(ros::Duration(0.1), timer_callback);
    
    ros::Timer timer2 = nh.createTimer(ros::Duration(0.1), timer_callback2);

    ros::Timer timer0 = nh.createTimer(ros::Duration(0.1), timer_callback0);

    ros::spin();
    return 0;
}