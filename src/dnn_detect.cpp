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
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;

enum model_t {
  MODEL_CAFFE,
  MODEL_YOLO
};

class DnnNode {
  private:
    ros::Publisher object_pub;

    image_transport::ImageTransport it;
    image_transport::Subscriber img_sub;

    // if set, we publish the images that contain objects
    bool publish_images;
    model_t model;
    bool swap_rb;

    int frame_num;
    float min_confidence;
    int im_size;
    float scale_factor;
    float mean_val;
    std::vector<std::string> class_names;

    image_transport::Publisher image_pub;

    cv::dnn::Net net;
    cv::Mat resized_image;

    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

  public:
    DnnNode(ros::NodeHandle &nh);
};


void DnnNode::imageCallback(const sensor_msgs::ImageConstPtr & msg)
{
    ROS_INFO("Got image %d", msg->header.seq);
    frame_num++;

    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        int w = cv_ptr->image.cols;
        int h = cv_ptr->image.rows;

        cv::resize(cv_ptr->image, resized_image, cvSize(im_size, im_size));
        cv::Mat blob = cv::dnn::blobFromImage(resized_image, scale_factor,
          cvSize(im_size, im_size), mean_val, swap_rb, false);

        net.setInput(blob, "data");

        cv::Mat detectionMat;
        if (model == MODEL_CAFFE) {
            cv::Mat objs = net.forward("detection_out");

            detectionMat = cv::Mat(objs.size[2], objs.size[3], CV_32F,
                                   objs.ptr<float>());
        }
        else if (model == MODEL_YOLO) {
            detectionMat = net.forward("detection_out");
        }

        for(int i = 0; i < detectionMat.rows; i++) {
            float confidence = 0.0f;
            int object_class = -1;

            if (model == MODEL_CAFFE) {
                confidence = detectionMat.at<float>(i, 2);
                object_class = (int)(detectionMat.at<float>(i, 1));
            }
            else if (model == MODEL_YOLO) {
 	        const int probability_index = 5;
                const int probability_size = detectionMat.cols - probability_index;
                float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

                object_class = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
                confidence = detectionMat.at<float>(i, (int)object_class + probability_index);
            }

            if (confidence >= min_confidence) {
                int x_min = 0;
                int x_max = 0;
                int y_min = 0;
                int y_max = 0;

                if (model == MODEL_CAFFE) {
                    x_min = static_cast<int>(detectionMat.at<float>(i, 3) * w);
                    y_min = static_cast<int>(detectionMat.at<float>(i, 4) * h);
                    x_max = static_cast<int>(detectionMat.at<float>(i, 5) * w);
                    y_max = static_cast<int>(detectionMat.at<float>(i, 6) * h);
                }
                else if (model == MODEL_YOLO) {
                    float x = detectionMat.at<float>(i, 0);
                    float y = detectionMat.at<float>(i, 1);
                    float width = detectionMat.at<float>(i, 2);
                    float height = detectionMat.at<float>(i, 3);
                    x_min = static_cast<int>((x - width / 2) * w);
                    y_min = static_cast<int>((y - height / 2) * h);
                    x_max = static_cast<int>((x + width / 2) * w);
                    y_max = static_cast<int>((y + height / 2) * h);
                }

                Rect object(x_min, y_min, y_max - x_min, y_max - y_min);

                rectangle(cv_ptr->image, object, Scalar(0, 255, 0));

                std::string class_name("unknown");
                if (object_class < class_names.size()) {
                    class_name = class_names[object_class];
                    std::string label = str(boost::format{"%1% %2%"} %
                                            class_name % confidence);

                    ROS_INFO("%s %f", label.c_str(), confidence);
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX,
                                                 0.5, 1, &baseLine);
                    rectangle(cv_ptr->image,
                              Rect(Point(x_min, y_min ),
                                   Size(labelSize.width, labelSize.height + baseLine)),
                              Scalar(255, 255, 255), CV_FILLED);
                    putText(cv_ptr->image, label,
                            Point(x_min, y_min+labelSize.height),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

                    dnn_detect::DetectedObject obj;
                    obj.header.frame_id = msg->header.frame_id;
                    obj.class_name = class_name;
                    obj.confidence = confidence;
                    obj.x_min = x_min;
                    obj.x_max = x_max;
                    obj.y_min = y_min;
                    obj.y_max = y_max;
                    object_pub.publish(obj);
                }
                else {
                    ROS_INFO("Unknown class %d conf %f", object_class, confidence);
                }
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
    frame_num = 0;

    std::string dir;
    std::string proto_net_file;
    std::string caffe_model_file;
    std::string yolo_weights_file;
    std::string yolo_cfg_file;

    std::string classes("background,"
       "aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,"
       "cow,diningtable,dog,horse,motorbike,person,pottedplant,"
       "sheep,sofa,train,tvmonitor");

    nh.param<bool>("publish_images", publish_images, false);
    nh.param<bool>("swap_rb", swap_rb, false);
    nh.param<string>("data_dir", dir, "");

    std::string model_string;
    nh.param<string>("model", model_string, "caffe");
    if (model_string == "caffe") {
        model = MODEL_CAFFE;
        nh.param<string>("protonet_file", proto_net_file,
                         "MobileNetSSD_deploy.prototxt.txt");
        nh.param<string>("caffe_model_file", caffe_model_file,
                         "MobileNetSSD_deploy.caffemodel");
    }
    else if (model_string == "yolo") {
        model = MODEL_YOLO;
        nh.param<string>("yolo_cfg_file", yolo_cfg_file,
                         "tiny-yolo.cfg");
        nh.param<string>("yolo_weights_file", yolo_weights_file,
                         "tiny-yolo.weights");
    }
    else {
        ROS_ERROR("Unknown model type %s", model_string.c_str());
        exit(1);
    }
    nh.param<float>("min_confidence", min_confidence, 0.12);
    nh.param<int>("im_size", im_size, 300);
    nh.param<float>("scale_factor", scale_factor, 0.007843f);
    nh.param<float>("mean_val", mean_val, 127.5f);
    nh.param<std::string>("class_names", classes, classes);

    boost::split(class_names, classes, boost::is_any_of(","));
    ROS_INFO("Read %d class names", (int)class_names.size());

    try {
        if (model == MODEL_CAFFE) {
            net = cv::dnn::readNetFromCaffe(dir + "/" + proto_net_file,
                                            dir + "/" + caffe_model_file);
        }
        else if (model == MODEL_YOLO) {
            net = cv::dnn::readNetFromDarknet(dir + "/tiny-yolo.cfg",
                                              dir + "/tiny-yolo.weights");
        }
    }
    catch(cv::Exception & e) {
        ROS_ERROR("cv exception: %s", e.what());
        exit(1);
    }

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
