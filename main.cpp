#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include <filesystem>
#include<math.h>

#define USE_CUDA true //use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;
namespace fs = std::filesystem;


class YoloApp {
public:
    YoloApp(const string& modelPath, const string& folderPath, const string& outputFolder)
        : modelPath_(modelPath), folderPath_(folderPath), outputFolder_(outputFolder), imageCount_(1) {}

    void run() {
        Yolo yolo;
        Net net;

        if (yolo.readModel(net, modelPath_, !USE_CUDA)) {
            cout << "Read net OK!" << endl;
        }
        else {
            cout << "Read ONNX model failed!";
            return;
        }

        vector<Scalar> color;
        srand(time(0));
        for (int i = 0; i < 80; i++) {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            color.push_back(Scalar(b, g, r));
        }

        bool continueProcessing = true;

        for (const auto& entry : fs::directory_iterator(folderPath_)) {
            const auto& filePath = entry.path();
            if (fs::is_regular_file(filePath) &&
                (filePath.extension() == ".jpg" || filePath.extension() == ".png" || filePath.extension() == ".jpeg")) {
                Mat img = imread(filePath.string());

                processImage(img, yolo, net, color);

                cout << "Press Enter to continue, Esc to exit, or any other key to display the prompt." << endl;
                char key = waitKey(0);

                if (key == 27) {  // 27 corresponds to the ASCII value of Esc key
                    continueProcessing = false;
                    break;
                }
                else if (key != 13) {  // 13 corresponds to the ASCII value of Enter key
                    cout << "Press Enter to continue, Esc to exit." << endl;
                    key = waitKey(0);
                    if (key == 27) {
                        continueProcessing = false;
                        break;
                    }
                }
            }

            if (!continueProcessing) {
                break;
            }
        }
    }

private:
    void processImage(Mat& img, Yolo& yolo, Net& net, const vector<Scalar>& color) {
        vector<Output> result;

        if (yolo.Detect(img, net, result)) {
            yolo.drawPred(img, result, color);

            // Save the processed image
            string outputPath = outputFolder_ + "/" + to_string(imageCount_) + ".jpg";
            imwrite(outputPath, img);
            cout << "Saved: " << outputPath << endl;
            imageCount_++;
        }
        else {
            cout << "Detect failed for image!" << endl;
        }

        namedWindow("Result of Detection", WINDOW_NORMAL);  // Create a resizable window
        imshow("Result of Detection", img);
        //destroyAllWindows(); // Close all previously created windows
    }

private:
    string modelPath_;
    string folderPath_;
    string outputFolder_;
    int imageCount_;
};

int main() {
#if (defined YOLOV5 && YOLOV5 == true)
    string model_path = "./models/yolov5n.onnx";
#else
    string model_path = "./models/best.onnx";
#endif

    string folder_path = "./images2"; // Change this to the path of your image folder

    string output_folder = "./output_images"; // Change this to the path where you want to save the processed images

    // Create the output folder if it doesn't exist
    fs::create_directory(output_folder);

    YoloApp yoloApp(model_path, folder_path, output_folder);
    yoloApp.run();
    return 0;
}
