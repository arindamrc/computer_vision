#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

class cEpipolar{
public:
    cEpipolar(){};
    ~cEpipolar(){};
    int loadImages();
    int loadCorresp();
    int displayCorresp();
    int computeFundMat();
    int drawEpipolar();
    int rectify();
    int applyHomography();

private:
    Mat             img1, img2;
    vector<Point2f> corr1, corr2;
};

int cEpipolar::loadImages(){
    img1 = imread("images/apt1.jpg");
    img2 = imread("images/apt2.jpg");
    if(0==img1.data || 0==img2.data){
        cout<<"error reading stereo images"<<endl;
        exit(-1);
    }
    return 0;
}

int cEpipolar::loadCorresp(){
    int numPts, dummy;

    ifstream iStream("images/corresp.txt");
    if(!iStream){
        cout<<"error reading the correspondence file"<<endl;
        exit(-1);
    }
    iStream >> numPts;
    iStream >> dummy;
    corr1.resize(numPts);
    corr2.resize(numPts);
    for(int idx=0; idx<numPts; ++idx){
        iStream >> corr1[idx].x;
        iStream >> corr1[idx].y;
        iStream >> corr2[idx].x;
        iStream >> corr2[idx].y;
    }
    return 0;
}

int cEpipolar::displayCorresp(){
    Mat i1, i2;
    img1.copyTo(i1);
    img2.copyTo(i2);

    cout<<"displaying corresponding points"<<endl;
    for(unsigned int idx=0; idx<corr1.size(); ++idx){
        circle(i1,corr1[idx],3,Scalar(255,0,0),2);
        circle(i2,corr2[idx],3,Scalar(255,0,0),2);
    }
    imshow("left_image",i1);
    imshow("right_image",i2);
    waitKey(0);
}

int cEpipolar::computeFundMat(){
    // implement the function
}

int cEpipolar::drawEpipolar(){
    // implement the function
}

int cEpipolar::rectify(){
    // implement the function
}

int cEpipolar::applyHomography(){
    // implement the function
}

class cDisparity{
public:
    cDisparity(){};
    ~cDisparity(){};
    int loadImages();
    int computeDisparity();
private:
    Mat img1, img2;
};

int cDisparity::loadImages(){
    img1 = imread("images/aloe1.png");
    img2 = imread("images/aloe2.png");
    if(0==img1.data || 0==img2.data){
        cout<<"error reading stereo images for disparity"<<endl;
        exit(-1);
    }
    return 0;
}

int cDisparity::computeDisparity(){
    return 0;
}

int main()
{
    // Q1: Fundamental Matrix
    cout<<"Q1 and Q2....."<<endl;
    cEpipolar epipolar;
    epipolar.loadImages();
    epipolar.loadCorresp();
    epipolar.displayCorresp();
    epipolar.computeFundMat();
    epipolar.drawEpipolar();

    // Q3 Disparity map
    cout<<endl<<endl<<"Q3....."<<endl;
    cDisparity disparity;
    disparity.loadImages();
    disparity.computeDisparity();

    // Q4: Rectifying image pair
    cout<<endl<<endl<<"Q4....."<<endl;
    epipolar.rectify();
    epipolar.applyHomography();

    return 0;
}
