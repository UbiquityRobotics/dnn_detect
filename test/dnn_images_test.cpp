#include <gtest/gtest.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "dnn_detect/DetectedObject.h"

class DnnImagesTest : public ::testing::Test {
protected:
  virtual void SetUp() { 
    it = new image_transport::ImageTransport(nh);
    image_pub = it->advertise("camera/image", 1);

    ros::NodeHandle nh_priv("~");
    nh_priv.getParam("image_directory", image_directory);
    object_sub = nh.subscribe("/dnn_objects", 1, &DnnImagesTest::object_callback, this);
    got_object = false;
  }

  virtual void TearDown() { delete it;}

  void publish_image(std::string file) {
    cv::Mat image = cv::imread(image_directory+file, CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    image_pub.publish(msg);
  }

  ros::NodeHandle nh;

  // Set up Publishing of static images
  image_transport::ImageTransport* it;
  image_transport::Publisher image_pub;

  bool got_object;
  ros::Subscriber object_sub;

  std::string image_directory;

  // Set up subscribing
  void object_callback(const dnn_detect::DetectedObject& f) {
    got_object = true;
  }
};


TEST_F(DnnImagesTest, cat) {
  ros::Rate loop_rate(5);
  while (nh.ok() && !got_object) {
    publish_image("cat.jpg");
    ros::spinOnce();
    loop_rate.sleep();
  }

}

int main(int argc, char** argv)
{

  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "DnnImagesTest");
  return RUN_ALL_TESTS();
}
