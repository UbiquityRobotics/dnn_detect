/*
 * Copyright (c) 2017, Ubiquity Robotics
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the FreeBSD Project.
 *
 */

#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include "dnn_detect/DetectedObject.h"
#include "dnn_detect/DetectedObjectArray.h"
#include "dnn_detect/Detect.h"

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;
using namespace cv;

std::condition_variable cond;
std::mutex mutx;

class DnnNode {
  private:
    ros::Publisher results_pub;

    image_transport::ImageTransport it;
    image_transport::Subscriber img_sub;

    // if set, we publish the images that contain objects
    bool publish_images;

    int frame_num;
    float min_confidence;
    int im_size;
    int rotate_flag;
    float scale_factor;
    float mean_val;
    std::vector<std::string> class_names;

    image_transport::Publisher image_pub;

    cv::dnn::Net net;
    cv::Mat resized_image;
    cv::Mat rotated_image;

    bool single_shot;
    volatile bool triggered;
    volatile bool processed;

    dnn_detect::DetectedObjectArray results;

    ros::ServiceServer detect_srv;

    bool trigger_callback(dnn_detect::Detect::Request &req,
                          dnn_detect::Detect::Response &res);

    void image_callback(const sensor_msgs::ImageConstPtr &msg);

  public:
    DnnNode(ros::NodeHandle &nh);
};

bool DnnNode::  trigger_callback(dnn_detect::Detect::Request &req,
                                 dnn_detect::Detect::Response &res)
{
    ROS_INFO("Got service request");
    triggered = true;

    std::unique_lock<std::mutex> lock(mutx);

    while (!processed) {
      cond.wait(lock);
    }
    res.result = results;
    processed = false;
    return true;
}


void DnnNode::image_callback(const sensor_msgs::ImageConstPtr & msg)
{
    if (single_shot && !triggered) {
        return;
    }
    triggered = false;

    ROS_INFO("Got image %d", msg->header.seq);
    frame_num++;

    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        int w = cv_ptr->image.cols;
        int h = cv_ptr->image.rows;

        if (rotate_flag >= 0) {
          cv::rotate(cv_ptr->image, rotated_image, rotate_flag);
          rotated_image.copyTo(cv_ptr->image);
        }

        cv::resize(cv_ptr->image, resized_image, cvSize(im_size, im_size));
        cv::Mat blob = cv::dnn::blobFromImage(resized_image, scale_factor,
          cvSize(im_size, im_size), mean_val, false);

        net.setInput(blob, "data");
        cv::Mat objs = net.forward("detection_out");

        cv::Mat detectionMat(objs.size[2], objs.size[3], CV_32F,
                             objs.ptr<float>());

        std::unique_lock<std::mutex> lock(mutx);
        results.header.frame_id = msg->header.frame_id;
        results.objects.clear();

        for(int i = 0; i < detectionMat.rows; i++) {

            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > min_confidence) {
                int object_class = (int)(detectionMat.at<float>(i, 1));

                int x_min = static_cast<int>(detectionMat.at<float>(i, 3) * w);
                int y_min = static_cast<int>(detectionMat.at<float>(i, 4) * h);
                int x_max = static_cast<int>(detectionMat.at<float>(i, 5) * w);
                int y_max = static_cast<int>(detectionMat.at<float>(i, 6) * h);

                std::string class_name;
                if (object_class >= class_names.size()) {
                     class_name = "unknown";
                     ROS_ERROR("Object class %d out of range of class names",
                               object_class);
                }
                else {
                     class_name = class_names[object_class];
                }
                std::string label = str(boost::format{"%1% %2%"} %
                                        class_name % confidence);

                ROS_INFO("%s", label.c_str());
                dnn_detect::DetectedObject obj;
                obj.class_name = class_name;
                obj.confidence = confidence;
                obj.x_min = x_min;
                obj.x_max = x_max;
                obj.y_min = y_min;
                obj.y_max = y_max;
                results.objects.push_back(obj);

                Rect object(x_min, y_min, x_max-x_min, y_max-y_min);

                rectangle(cv_ptr->image, object, Scalar(0, 255, 0));
                int baseline=0;
                cv::Size text_size = cv::getTextSize(label,
                                     FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseline);
                putText(cv_ptr->image, label, Point(x_min, y_min-text_size.height),
                        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0));
            }
        }

        results_pub.publish(results);

	image_pub.publish(cv_ptr->toImageMsg());

    }
    catch(cv_bridge::Exception & e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    catch(cv::Exception & e) {
        ROS_ERROR("cv exception: %s", e.what());
    }
    ROS_DEBUG("Notifying condition variable");
    processed = true;
    cond.notify_all();
}

DnnNode::DnnNode(ros::NodeHandle & nh) : it(nh)
{
    frame_num = 0;

    std::string dir;
    std::string proto_net_file;
    std::string caffe_model_file;
    std::string classes("background,"
       "aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,"
       "cow,diningtable,dog,horse,motorbike,person,pottedplant,"
       "sheep,sofa,train,tvmonitor");

    nh.param<bool>("single_shot", single_shot, false);

    nh.param<bool>("publish_images", publish_images, false);
    nh.param<string>("data_dir", dir, "");
    nh.param<string>("protonet_file", proto_net_file,
                     "MobileNetSSD_deploy.prototxt.txt");
    nh.param<string>("caffe_model_file", caffe_model_file,
                     "MobileNetSSD_deploy.caffemodel");
    nh.param<float>("min_confidence", min_confidence, 0.2);
    nh.param<int>("im_size", im_size, 300);
    nh.param<int>("rotate_flag", rotate_flag, -1);
    nh.param<float>("scale_factor", scale_factor, 0.007843f);
    nh.param<float>("mean_val", mean_val, 127.5f);
    nh.param<std::string>("class_names", classes, classes);

    boost::split(class_names, classes, boost::is_any_of(","));
    ROS_INFO("Read %d class names", (int)class_names.size());

    try {
        net = cv::dnn::readNetFromCaffe(dir + "/" + proto_net_file,
                                        dir + "/" + caffe_model_file);
    }
    catch(cv::Exception & e) {
        ROS_ERROR("cv exception: %s", e.what());
        exit(1);
    }

    triggered = false;

    detect_srv = nh.advertiseService("detect", &DnnNode::trigger_callback, this);

    results_pub =
        nh.advertise<dnn_detect::DetectedObjectArray>("/dnn_objects", 20);

    image_pub = it.advertise("/dnn_images", 1);

    img_sub = it.subscribe("/camera", 1,
                           &DnnNode::image_callback, this);

    ROS_INFO("DNN detection ready");
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "dnn_detect");
    ros::NodeHandle nh("~");

    DnnNode node = DnnNode(nh);
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}
