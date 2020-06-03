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
    return imread(filename, IMREAD_GRAYSCALE/*, CV_8UC1*/);
}

int main(int argc, char** argv){

    vector<Mat> x_train, x_test;
    vector<int> y_train, y_test;

    /*************** LOAD TRAIN DATASET ***************/

    cout << "Lectura del conjunto train..." << endl;

    vector<cv::String> train_sujeto_0_file_names, train_sujeto_1_file_names;
    glob("../dataset/train/sujeto_0/*.jpg", train_sujeto_0_file_names, false);
    glob("../dataset/train/sujeto_1/*.jpg", train_sujeto_1_file_names, false);

    for(int i = 0; i < train_sujeto_0_file_names.size(); i++){
		Mat img = read_image(train_sujeto_0_file_names[i]);
		x_train.push_back(img);
		y_train.push_back(0);
    }

	for(int i = 0; i < train_sujeto_1_file_names.size(); i++){
		Mat img = read_image(train_sujeto_1_file_names[i]);
		x_train.push_back(img);
		y_train.push_back(1);
	}

    /*************** LOAD TEST DATASET ***************/

    cout << "Lectura del conjunto test..." << endl;

    vector<cv::String> test_sujeto_0_file_names, test_sujeto_1_file_names;
    glob("../dataset/test/sujeto_0/*.jpg", test_sujeto_0_file_names, false);
    glob("../dataset/test/sujeto_1/*.jpg", test_sujeto_1_file_names, false);

    vector<cv::String> test_unknown_file_names;
    glob("../dataset/test/unknown/*.jpg", test_unknown_file_names, false);

    for(int i = 0; i < test_unknown_file_names.size(); i++){
        Mat img = read_image(test_unknown_file_names[i]);
        x_test.push_back(img);
        y_test.push_back(-1);
    }

    for(int i = 0; i < test_sujeto_0_file_names.size(); i++){
		Mat img = read_image(test_sujeto_0_file_names[i]);
		x_test.push_back(img);
		y_test.push_back(0);
    }

	for(int i = 0; i < test_sujeto_1_file_names.size(); i++){
		Mat img = read_image(test_sujeto_1_file_names[i]);
		x_test.push_back(img);
		y_test.push_back(1);
	}


    /*************** TRAIN FACE ROCOGNIZER ***************/

    cout << "Entrenando modelo..." << endl;

    Ptr<FaceRecognizer> fisher_model = createFisherFaceRecognizer();
    fisher_model->setThreshold(100.0);

    Ptr<FaceRecognizer> lbphf_model = createLBPHFaceRecognizer();
    lbphf_model->setThreshold(70.0);

    Ptr<FaceRecognizer> eigen_model = createEigenFaceRecognizer();
    eigen_model->setThreshold(4000.0);

    vector< Ptr<FaceRecognizer> > models;
    models.push_back(fisher_model);
    models.push_back(lbphf_model);
    models.push_back(eigen_model);

    vector<String> model_names;
    model_names.push_back("fisher");
    model_names.push_back("lbphf");
    model_names.push_back("eigen");

    for(int i = 0; i < models.size(); i++){
        models[i]->train(x_train, y_train);
        cout << "Guardando modelo " << model_names[i] << "..." << endl;
        models[i]->save("../model/" + model_names[i] + "-model.yml");
    }

    /****************************** VALIDATE MODELS ******************************/

    vector< vector<int> > all_predictions(x_test.size(), vector<int>(3,0));

    for(int k = 0; k < models.size(); k++){

        cout << "Realizando predicciones con modelo " << model_names[k] << endl;

        vector<int> predictions;

        vector< vector<int> > confusion_matrix(3, vector<int>(3, 0));

        for(int i = 0; i < x_test.size(); i++){
            int prediction;
            double confidence;

            models[k]->predict(x_test[i], prediction, confidence);

            predictions.push_back(prediction);
            all_predictions[i][k] = prediction;

            confusion_matrix[ y_test[i]+1 ][ prediction+1 ]++;
        }

        double accuracy = 0.0;

        for(int i = 0; i < confusion_matrix.size(); i++){
            accuracy += confusion_matrix[i][i];
        }

        accuracy /= x_test.size();

        cout << "Accuracy: " << accuracy << "%" << endl;

        cout << "   -1   0   1" << endl;

        for(int i = 0; i < 3; i++){
            cout << i-1 << " " << confusion_matrix[i][0] << " ";
            cout << confusion_matrix[i][1] << " ";
            cout << confusion_matrix[i][2] << endl;
        }
    }

    vector< vector<int> > confusion_matrix(3, vector<int>(3, 0));

    for(int i = 0; i < x_test.size(); i++){
        vector<int> votes(3,0);

        for(int j = 0; j < 3; j++){
            votes[all_predictions[i][j]+1]++;
        }

        bool win_by_majority = false;

        for(int j = 0; j < 3; j++){
            if(votes[j] == 2 || votes[j] == 3){
                confusion_matrix[ y_test[i]+1 ][ j ]++;
                win_by_majority = true;
            }
        }

        if(!win_by_majority){
            confusion_matrix[ y_test[i]+1 ][ all_predictions[i][1]+1 ]++;
        }
    }

    double accuracy = 0.0;

    for(int i = 0; i < confusion_matrix.size(); i++){
        accuracy += confusion_matrix[i][i];
    }

    accuracy /= x_test.size();

    cout << "Accuracy: " << accuracy << "%" << endl;

    cout << "   -1   0   1" << endl;

    for(int i = 0; i < 3; i++){
        cout << i-1 << " " << confusion_matrix[i][0] << " ";
        cout << confusion_matrix[i][1] << " ";
        cout << confusion_matrix[i][2] << endl;
    }

    /*********************************************************************/

    return 0;
}
