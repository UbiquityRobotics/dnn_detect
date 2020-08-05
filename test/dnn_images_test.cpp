#include <gtest/gtest.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "dnn_detect/DetectedObject.h"
#include "dnn_detect/DetectedObjectArray.h"
#include "dnn_detect/Detect.h"

#include <boost/thread/thread.hpp>

#if CV_MAJOR_VERSION < 4
    #define IMREAD_COLOR_MODE CV_LOAD_IMAGE_COLOR
#else
    #define IMREAD_COLOR_MODE cv::IMREAD_COLOR
#endif

class DnnImagesTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    it = new image_transport::ImageTransport(nh);
    image_pub = it->advertise("camera/image", 1);

    ros::NodeHandle nh_priv("~");
    nh_priv.getParam("image_directory", image_directory);
    object_sub = nh.subscribe("/dnn_objects", 1, &DnnImagesTest::object_callback, this);
    got_object = false;
    got_cat = false;

  }

  // Make a service request to trigger detection
  void trigger() {
    ros::NodeHandle node;
    ros::ServiceClient client =
       node.serviceClient<dnn_detect::Detect>("/dnn_detect/detect");
    dnn_detect::Detect d;
    client.call(d);
  }

  virtual void TearDown() { delete it;}

  void publish_image(std::string file) {
    boost::thread trig(&DnnImagesTest::trigger, this);

    sleep(1);
    cv::Mat image = cv::imread(image_directory+file, IMREAD_COLOR_MODE);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8",
        image).toImageMsg();
    image_pub.publish(msg);
  }

  ros::NodeHandle nh;

  // Set up Publishing of static images
  image_transport::ImageTransport* it;
  image_transport::Publisher image_pub;

  bool got_object;
  bool got_cat;
  ros::Subscriber object_sub;

  std::string image_directory;

  // Set up subscribing
  void object_callback(const dnn_detect::DetectedObjectArray& results) {
    got_object = true;
    for (const auto& obj : results.objects) {
      if (obj.class_name == "cat") {
        got_cat = true;
      }
    }
  }
};


TEST_F(DnnImagesTest, cat) {
  ros::Rate loop_rate(5);
  while (nh.ok() && !got_object && !got_cat) {
    publish_image("cat.jpg");
    ros::spinOnce();
    loop_rate.sleep();
  }

  ASSERT_TRUE(got_cat);
}

int main(int argc, char** argv)
{

  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "DnnImagesTest");
  return RUN_ALL_TESTS();
}
