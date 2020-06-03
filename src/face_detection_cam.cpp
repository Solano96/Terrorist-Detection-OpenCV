#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include <opencv2/face/facerec.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;

Mat read_image(String filename){
    return imread(filename);
}

void display_image(Mat im, String windowName = ""){
    cout << "Display" << endl;
    namedWindow(windowName);
    imshow("Face Detection", im);
    waitKey(0);
    destroyWindow(windowName);
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, vector< Ptr<FaceRecognizer> > models){
    vector<Rect> faces;
    Mat gray;
    int tam = 128;
    Scalar color;
    Size size(tam,tam);
    cvtColor( img, gray, COLOR_BGR2GRAY );

    cascade.detectMultiScale( gray, faces, scale, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    #pragma omp parallel for
    for ( size_t i = 0; i < faces.size(); i++ ){
        Rect r = faces[i];
        Mat gray_img = img(r);
        cvtColor(img(r), gray_img, CV_BGR2GRAY);
        resize(gray_img, gray_img, size);

        double confidence [3] = {0.0, 0.0, 0.0};
        int predicted [3] = {-1,-1,-1};
        vector<int> votes(3,0);
        int result = 0;

        for(int j = 0; j < models.size(); j++){
            models[j]->predict(gray_img, predicted[j], confidence[j]);
            votes[predicted[j]+1]++;
        }


        // Sistema de votacion

        int winner = -1;

        for(int j = 0; j < 3; j++){
            if(votes[j] == 2 || votes[j] == 3){
                winner = j;
            }
        }

        if(winner == -1){
            winner = 1;
        }

        if(winner == 0){
            color = Scalar(0, 255, 0);
        }
        else{
            color = Scalar(0, 0, 255);
        }
		// Draw rectangle on the image
        int x = cvRound(r.x);
        int y = cvRound(r.y);
        int width = cvRound(r.x + r.width);
        int height = cvRound(r.y + r.height);
        rectangle( img, cvPoint(x, y), cvPoint(width, height), color, 2, 8, 0);
    }
}

int main(int argc, char** argv){
    // Load the cascade classifier
    CascadeClassifier cascade;
    cascade.load( "./cascade.xml" ) ;
    double scale=1.1;
    bool stop_capturing = false;

    // Load models

    cout << "Load models..." << endl;

    Ptr<FaceRecognizer> fisher_model = createFisherFaceRecognizer();
    fisher_model->setThreshold(200.0);

    Ptr<FaceRecognizer> lbphf_model = createLBPHFaceRecognizer();
    lbphf_model->setThreshold(100.0);

    Ptr<FaceRecognizer> eigen_model = createEigenFaceRecognizer();
    eigen_model->setThreshold(4500.0);

	fisher_model->load("../model/fisher-model.yml");
    lbphf_model->load("../model/lbphf-model.yml");
    eigen_model->load("../model/eigen-model.yml");

    vector< Ptr<FaceRecognizer> > models;
    models.push_back(fisher_model);
    models.push_back(lbphf_model);
    models.push_back(eigen_model);

    VideoCapture cap;

    // open the default camera
    if(!cap.open(0)){
		return 0;
	}

    while(!stop_capturing){
          Mat frame;
          cap >> frame;

		  if(frame.empty() || waitKey(10) == 27){
			  stop_capturing = true;
		  }
		  else{
			  detectAndDraw(frame, cascade, scale, models);
	          imshow("Face Detection", frame);
		  }
    }

    return 0;
}
