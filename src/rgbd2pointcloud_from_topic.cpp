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
bool show_img;
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

void rgbd_callback(const sensor_msgs::Image::ConstPtr& rgb_msg,const sensor_msgs::Image::ConstPtr& depth_msg){
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(rgb_msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(depth_msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat  rgb_img = cv_ptrRGB->image.clone();
    cv::Mat  depth_img =  cv_ptrD->image.clone();
    img_header = rgb_msg->header;
    generatePointCloud(rgb_img,depth_img);
    begin_pub = true;
    if(show_img){
        cv::imshow("rgbd2pt rgb_img",rgb_img);
        cv::imshow("rgbd2pt depth_img",depth_img);
        cv::waitKey(10);
    }

}

void camera_pose_callback(const nav_msgs::Odometry::ConstPtr &odom_msg){
    Eigen::Quaterniond qq = Eigen::Quaterniond( 
        odom_msg->pose.pose.orientation.w,
        odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y,
        odom_msg->pose.pose.orientation.z);
    camera_R = Eigen::Matrix3d(qq);
    camera_t = Eigen::Vector3d( 
        odom_msg->pose.pose.position.x,
        odom_msg->pose.pose.position.y,
        odom_msg->pose.pose.position.z); 
    camera_time = odom_msg->header.stamp.toSec();

    generateOdomPath(camera_R,camera_t);
    
}

void camera_tf_callback(const nav_msgs::Odometry::ConstPtr &odom_msg){
    if(pointcloud_mode ==1){
    // Quaternion(const tfScalar& x, const tfScalar& y, const tfScalar& z, const tfScalar& w) 
        tf::Quaternion tf_camera_rotation(odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y,
        odom_msg->pose.pose.orientation.z,
        odom_msg->pose.pose.orientation.w);
        tf::Vector3 tf_camera_translation (odom_msg->pose.pose.position.x,odom_msg->pose.pose.position.y,odom_msg->pose.pose.position.z);
        point_cloud_tf = tf::Transform (tf_camera_rotation, tf_camera_translation);
        camera_odom_header = odom_msg->header;

        Eigen::Quaterniond qq(odom_msg->pose.pose.orientation.w,
                                                        odom_msg->pose.pose.orientation.x,
                                                        odom_msg->pose.pose.orientation.y,
                                                        odom_msg->pose.pose.orientation.z);
        Eigen::Matrix3d RR(qq);
        Eigen::Vector3d tt(odom_msg->pose.pose.position.x,odom_msg->pose.pose.position.y,odom_msg->pose.pose.position.z);
        generateOdomPath(RR,tt);
    }
}

void timer_callback(const ros::TimerEvent& e)
{
    if(!begin_pub){
        return;
    }
    ROS_INFO("pointcloud published");
    std::cout<<"header "<<point_cloud_msg.header.frame_id<<std::endl;
    pub_pointcloud.publish(point_cloud_msg);
    pub_path.publish(path_msg);
    pub_odom.publish(odom_msg);
}

void timer_callback2(const ros::TimerEvent& e)
{
    if(pointcloud_mode == 2){
        return;
    }
    if(pointcloud_mode == 3){
        return;
    }

    if(!begin_pub){
        return;
    }
    static tf::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf::StampedTransform(point_cloud_tf, point_cloud_header.stamp, "world", point_cloud_header.frame_id));
}


int main(int argc, char **argv)
{
    camera_factor = 1000.0;
    camera_cx = 319.16667152289227;
    camera_cy = 235.58360480225772;
    camera_fx = 609.70550296798035;
    camera_fy = 609.09579671294716;

    // pointcloud_mode = 0; //pointclond on the origin point
    // pointcloud_mode = 1; //pointcloud by tf
    // pointcloud_mode = 2; //point clood by pose
    // pointcloud_mode = 3; //no tf only point
    
    ros::init(argc, argv, "ego_estimator");
    ros::NodeHandle nh("~");


    std::string camera_odom_name = "/odom";

    nh.param("/rgbd2pointcloud/pointcloud_mode", pointcloud_mode, pointcloud_mode);
    nh.param("/rgbd2pointcloud/camera_odom_name", camera_odom_name, camera_odom_name);
    nh.param("/rgbd2pointcloud/show_img", show_img, show_img);

    std::cout<<"pointcloud_mode "<<pointcloud_mode<<std::endl;
    std::cout<<"camera_odom_name "<<camera_odom_name<<std::endl;
    std::cout<<"show_img "<<show_img<<std::endl;

    {
        tf::Quaternion tf_camera_rotation(0,0,0,1);
        tf::Vector3 tf_camera_translation (0,0,0);
        point_cloud_tf = tf::Transform (tf_camera_rotation, tf_camera_translation);
        camera_odom_header.frame_id = "world";
    }

    path_msg.header.frame_id = "world";
    odom_msg.header.frame_id = "world";

    pub_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/visualization_point_cloud", 100);
    pub_path = nh.advertise<nav_msgs::Path>("/visualization_path", 100);
    pub_odom = nh.advertise<nav_msgs::Odometry>("/visualization_odom", 100);

    ros::Timer timer1 = nh.createTimer(ros::Duration(0.1), timer_callback);
    
    ros::Timer timer2 = nh.createTimer(ros::Duration(0.01), timer_callback2);

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync_img(sync_pol(10), rgb_sub,depth_sub);
    sync_img.registerCallback(boost::bind(&rgbd_callback,_1,_2));

    ros::Subscriber sub_wheel_odom;

    if(pointcloud_mode ==2){
        sub_wheel_odom = nh.subscribe(camera_odom_name, 200, camera_pose_callback);
    }else if(pointcloud_mode ==1||pointcloud_mode ==0){
        sub_wheel_odom = nh.subscribe(camera_odom_name, 200, camera_tf_callback);
    }

    ros::spin();
    return 0;
}