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

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <string>

using namespace std;
using namespace cv;

const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const char* classNames[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

class DnnNode {
  private:
    ros::Publisher object_pub;

    image_transport::ImageTransport it;
    image_transport::Subscriber img_sub;

    // if set, we publish the images that contain objects
    bool publish_images;

    int frameNum;
    float min_confidence;

    image_transport::Publisher image_pub;

    cv::dnn::Net net;

    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

  public:
    DnnNode(ros::NodeHandle &nh);
};


void DnnNode::imageCallback(const sensor_msgs::ImageConstPtr & msg) {
    ROS_INFO("Got image %d", msg->header.seq);
    frameNum++;

    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        int w = cv_ptr->image.cols;
        int h = cv_ptr->image.rows;

        cv::Mat resized_image;
                imwrite("cat2.jpg", cv_ptr->image);
        cv::resize(cv_ptr->image, resized_image, cvSize(300, 300));
        cv::Mat blob = cv::dnn::blobFromImage(resized_image, 0.007843, cvSize(300, 300), 127.5, false);

        net.setInput(blob, "data");
        cv::Mat objs = net.forward("detection_out");

        cv::Mat detectionMat(objs.size[2], objs.size[3], CV_32F, objs.ptr<float>());

        for(int i = 0; i < detectionMat.rows; i++) {
 
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > min_confidence) {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                int x_min = static_cast<int>(detectionMat.at<float>(i, 3) * w);
                int y_min = static_cast<int>(detectionMat.at<float>(i, 4) * h);
                int x_max = static_cast<int>(detectionMat.at<float>(i, 5) * w);
                int y_max = static_cast<int>(detectionMat.at<float>(i, 6) * h);

                std::string label = str(boost::format{"%1% %2%"} % 
                                        classNames[objectClass] % confidence);

                dnn_detect::DetectedObject obj;
                obj.header.frame_id = msg->header.frame_id;
                obj.class_name = classNames[objectClass];
                obj.confidence = confidence;
                obj.x_min = x_min;
                obj.x_max = x_max;
                obj.y_min = y_min;
                obj.y_max = y_max;
                object_pub.publish(obj);

                Rect object(x_min, y_min, x_max-x_min, y_max-y_min);

                rectangle(cv_ptr->image, object, Scalar(0, 255, 0));
                int baseline=0;
                cv::Size text_size = cv::getTextSize(label, 
                                     FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseline);
                putText(cv_ptr->image, label, Point(x_min, y_min-text_size.height),
                        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0));
                
            }
        }

	image_pub.publish(cv_ptr->toImageMsg());
    }
    catch(cv_bridge::Exception & e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    catch(cv::Exception & e) {
        ROS_ERROR("cv exception: %s", e.what());
    }
}

DnnNode::DnnNode(ros::NodeHandle & nh) : it(nh)
{
    frameNum = 0;

    std::string dir;
    std::string proto_net_file;
    std::string caffe_model_file;

    nh.param<string>("data_dir", dir, "");
    nh.param<string>("protonet_file", proto_net_file, "MobileNetSSD_deploy.prototxt.txt");
    nh.param<string>("caffe_model_file", caffe_model_file, "MobileNetSSD_deploy.caffemodel");
    nh.param<float>("min_confidence", min_confidence, 0.2);

    try {
        net = cv::dnn::readNetFromCaffe(dir + "/" + proto_net_file, dir + "/" + caffe_model_file);
    }
    catch(cv::Exception & e) {
        ROS_ERROR("cv exception: %s", e.what());
        exit(1);
        
    }

    nh.param<bool>("publish_images", publish_images, false);
    image_pub = it.advertise("/dnn_images", 1);
    object_pub =  nh.advertise<dnn_detect::DetectedObject>("/dnn_objects", 20); 
    img_sub = it.subscribe("/camera", 1,
                           &DnnNode::imageCallback, this);

    ROS_INFO("DNN detection ready");
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "dnn_detect");
    ros::NodeHandle nh("~");

    DnnNode node = DnnNode(nh);

    ros::spin();

    return 0;
}
