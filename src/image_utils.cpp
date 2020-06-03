#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using namespace cv;
using namespace std;


Mat read_image(String filename){
    return imread(filename/*, CV_8UC1*/);
}


void display_image(Mat im, String windowName = ""){
    cout << "Display" << endl;
    namedWindow(windowName);
    imshow("Face Detection", im);
    waitKey(0);
    destroyWindow(windowName);
}


void resize_images(String path_images){
    vector<cv::String> file_names;
    String images_to_find = path_images + "/*.*";

    glob(images_to_find, file_names, false);

    for(int i = 0; i < file_names.size(); i++){
        String file_name = file_names[i];
        cout << file_name << endl;
        Mat img = read_image(file_name);
        int tam = 128;
        Size size(tam,tam);//the dst image size,e.g.100x100
        Mat resizeImg;
        resize(img, resizeImg, size);//resize image
        imwrite("../temporal/a" + to_string(i) + ".jpg", resizeImg);
    }
}


void get_and_save_faces( Mat& img, CascadeClassifier& cascade, double scale, int &num_image){
    vector<Rect> faces;
    Mat gray;
    cvtColor( img, gray, COLOR_BGR2GRAY );
    cascade.detectMultiScale( gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ ){
        Rect r = faces[i];
        String file_name = "../temporal/a" + to_string(num_image) + ".jpg";
        Mat img_cut = img(r);
        int tam = 128;
        Size size(tam,tam);//the dst image size,e.g.100x100
        Mat resizeImg;
        resize(img_cut, resizeImg, size);

        imwrite(file_name, resizeImg );
        num_image++;
    }
}


void get_faces_from_images(String path_images){
    // Load the cascade classifier
    CascadeClassifier cascade;
    cascade.load( "./cascade.xml" ) ;
    double scale=1;

    vector<cv::String> file_names;
    glob(path_images+ "/*.*", file_names, false);
    int num_image = 0;

    for(int i = 0; i < file_names.size(); i++){
        Mat img = read_image(file_names[i]);
        get_and_save_faces(img, cascade, scale, num_image);
    }
}


int main(int argc, char** argv){

    if(strcmp(argv[1], "-r") == 0){
        String path_images = argv[2];
        resize_images(path_images);
    }
    else if(strcmp(argv[1], "-f") == 0){
        String path_images = argv[2];
        get_faces_from_images(path_images);
    }

    return 0;
}
