#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define EPSILON 10e-6

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
    Mat img1, img2;
    Mat fundamentalMatrix;
    RNG rng;
    vector<Point2f> corr1, corr2;
    Mat Hp1, Hp2, Hr1, Hr2;
    Point2f findCentroid(vector<Point2f>& points);
    float findScale(vector<Point2f>& points, Point2f& c);
    Mat_<float> getHomogeneous(Point2f& pt);
    void drawLine(Mat_<float>& p, Mat& l, Mat& img);
    void drawLine2(Mat_<float>& p, Mat& l, Mat& img);
    void displayPoints(Mat& i1, Mat& i2);
    Mat findNullSpace(Mat& m);
    Mat_<float> getCrossRepresentation(Mat& m);
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
    this->displayPoints(i1, i2);
}

void cEpipolar::displayPoints(cv::Mat& i1, cv::Mat& i2)
{
        for(unsigned int idx=0; idx<corr1.size(); ++idx){
                circle(i1,corr1[idx],3,Scalar(255,0,0),2);
                circle(i2,corr2[idx],3,Scalar(255,0,0),2);
        }
        imshow("left_image",i1);
        imshow("right_image",i2);
        waitKey(0);
}


// compute the centroid of the list of points
Point2f cEpipolar::findCentroid(vector<Point2f>& points)
{
        int n = points.size();
        Point2f avg;
        for(int i = 0; i < n; i++){
                avg.x += points[i].x;
                avg.y += points[i].y;
        }
        avg.x /= n;
        avg.y /= n;
        
        return avg;
}

// Compute the normalization scale of the list of points
float cEpipolar::findScale(vector<Point2f>& points, Point2f& c)
{
        float rms = 0.0f;
        int n = points.size();
        for(int i = 0; i < n; i++){
                float x = points[i].x - c.x;
                float y = points[i].y - c.y;
                rms+= sqrt(x * x + y * y);
        }
        rms /= n;
        double scale = pow(2, 0.5)/rms;
        
        return scale;
}

// Returns the homogeneous representation of a point
Mat_<float> cEpipolar::getHomogeneous(Point2f& pt)
{
        Mat_<float> p(3,1);  
        p.setTo(1.0f);

        p(0,0) = pt.x;
        p(1,0) = pt.y;
        
        return p;
}



// Compute the Fundamental Matrix using the 8-point algorithm
int cEpipolar::computeFundMat(){
        int n = this->corr1.size();
        
        // calculate centroids
        Point2f c1 = this->findCentroid(this->corr1);
        Point2f c2 = this->findCentroid(this->corr2);
        
        // calculate points
        float s1 = this->findScale(this->corr1, c1);
        float s2 = this->findScale(this->corr2, c2);
        
        // rescale the centroids too!
        c1 *= s1;
        c2 *= s2;
        
        // Calculate transformations
        Mat_<float> t1(3,3);
        Mat_<float> t2(3,3);
        t1.setTo(0.0f);
        t2.setTo(0.0f);
        
        t1(0,0) = s1;
        t1(1,1) = s1;
        t1(0,2) = -c1.x;
        t1(1,2) = -c1.y;
        t1(2,2) = 1.0f;
        
        t2(0,0) = s2;
        t2(1,1) = s2;
        t2(0,2) = -c2.x;
        t2(1,2) = -c2.y;
        t2(2,2) = 1.0f;
        
        // Allocate matrix A
        Mat_<float> A(n, 9);
        A.setTo(1.0f);
        
        for(int i = 0; i < n; i++){
                // Convert to homogeneous
                Mat_<float> p1 = this->getHomogeneous(corr1[i]);
                Mat_<float> p2 = this->getHomogeneous(corr2[i]);

                // Apply transformation
                p1 = t1*p1;
                p2 = t2*p2;
              
                // Set up A
                A(i,0) = p2(0,0)*p1(0,0);
                A(i,1) = p2(0,0)*p1(1,0);
                A(i,2) = p2(0,0);
                A(i,3) = p2(1,0)*p1(0,0);
                A(i,4) = p2(1,0)*p1(1,0);
                A(i,5) = p2(1,0);
                A(i,6) = p1(0,0);
                A(i,7) = p1(1,0);
        }
        
        // Calculate f by SVD
        Mat A_u, A_vt, A_w;
        SVDecomp(A, A_w, A_u, A_vt);
        Mat v = A_vt.t();
        Mat f = v.col(v.cols - 1).clone(); // Take last col of v corresponding to the least singular value
        
        assert(f.rows == 9);
        std::cout << "f: " << f.size() << std::endl;
        
        Mat F = f.reshape(1,3);
        
        // Enforce the singularity constraint
        Mat F_u, F_vt, F_w_col;
        SVDecomp(F, F_w_col, F_u, F_vt);
        
        assert(F_w_col.rows == 3);
        
        Mat_<float> F_w(3,3);
        F_w.setTo(0.0f);
        
        F_w(0,0) = F_w_col.at<float>(0,0);
        F_w(1,1) = F_w_col.at<float>(1,0);
        
        std::cout << "F_u: " << F_u.size() << ", F_vt: " << F_vt.size() << ", F_w: " << F_w_col.size() << std::endl;
        Mat F_ = F_u * F_w * F_vt;
        std::cout << "F_: " << F_ << std::endl;
        
        F = t2.t() * F_ * t1;
        std::cout << "F: " << F << std::endl;
        this->fundamentalMatrix = F;
}

// Helper function. Draw the given epipolar line on the image
void cEpipolar::drawLine(Mat_<float>& p, Mat& l, Mat& img)
{
        float a = l.at<float>(0,0);
        float b = l.at<float>(1,0);
        float c = l.at<float>(2,0);
        float u = p(0,0);
        float v = p(1,0);
        
        const float LENGTH = img.rows;
        
        Point2f e1, e2; // line endpoints
        
        // Drawing an epipolar line. From: http://www2.ece.ohio-state.edu/~aleix/MultipleImages.pdf
        if(fabs(a) < fabs(b)){
                float d = LENGTH / pow(((a/b) * (a/b)) + 1, 0.5);
                float f1 = (-c - a * (u - d) ) / b;
                float f2 = (-c - a * (u + d) ) / b;
                e1.x = u - d;
                e1.y = f1;
                e2.x = u + d;
                e2.y = f2;
        }else{
                float d = LENGTH / pow(((b/a) * (b/a)) + 1, 0.5);
                float f1 = (-c - b * (v - d)) / a;
                float f2 = (-c - b * (v + d)) / a;
                e1.x = f1;
                e1.y = v - d;
                e2.x = f2;
                e2.x = v + d;
        }
        Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
        line(img,e1,e2,color,2);
}

void cEpipolar::drawLine2(Mat_<float>& p, cv::Mat& l, cv::Mat& img)
{
        float x0 = 0, xn = img.cols - 1;
        
        // ax + by + c = 0 => y = (-c - ax) / b
        float a = l.at<float>(0,0);
        float b = l.at<float>(1,0);
        float c = l.at<float>(2,0);
        
        Point2f e1, e2; // line endpoints
        e1.x = x0;
        e1.y = (-c - (a * x0)) / b;
        e2.x = xn;
        e2.y = (-c - (a * xn)) / b;
        
        Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
        line(img,e1,e2,color,1);
}



int cEpipolar::drawEpipolar(){
        int n = this->corr1.size(); // no. of corresponding points
        Mat F = this->fundamentalMatrix;
        
        Mat i1 = this->img1.clone();
        Mat i2 = this->img2.clone();
        
        for(int i = 0; i < n; i++){
               Mat_<float> p1 = this->getHomogeneous(this->corr1[i]); 
               Mat_<float> p2 = this->getHomogeneous(this->corr2[i]); 
               
               // Get the epipolar lines of the respective images
               Mat l2 = F * p1;
               Mat l1 = F.t() * p2;
               this->drawLine2(p1, l2, i1);
               this->drawLine2(p2, l1, i2);
        }
        this->displayPoints(i1, i2);
}

Mat cEpipolar::findNullSpace(Mat& m)
{
        assert (fabs(determinant(m)) < EPSILON);
        Mat u, w, vt, N;
        SVDecomp(m, w, u, vt);
        int i = 0;
        bool found = false;
        for(i = 0; i < w.rows; i++){
                if(fabs(w.at<float>(i,0)) < EPSILON){
                     found = true;
                     break;
                }
        }
        if(found){
                Mat v = vt.t();
                N = v.col(i).clone();
        }
        return N;
}

Mat_<float> cEpipolar::getCrossRepresentation(Mat& m)
{
        Mat_<float> c(3,3);
        c.setTo(0.0f);
        c(0,1) = -m.at<float>(2,0);
        c(0,2) = m.at<float>(1,0);
        c(1,0) = m.at<float>(2,0);
        c(1,2) = -m.at<float>(0,0);
        c(2,0) = -m.at<float>(1,0);
        c(2,1) = m.at<float>(0,0);
        
        return c;
}



int cEpipolar::rectify(){
        // Find the epipoles for both images
        Mat F = this->fundamentalMatrix;
        Mat e1 = this->findNullSpace(F);
        e1 /= e1.at<float>(2,0);
        Mat e1_x = this->getCrossRepresentation(e1);
        
        Mat F_t = F.t();
        
        // Find matrices A and B
        int w1 = this->img1.cols;
        int h1 = this->img1.rows;
        int w2 = this->img2.cols;
        int h2 = this->img2.rows;
        
        Mat_<float> PP_t_1(3,3);
        PP_t_1.setTo(0.0f);
        PP_t_1(0,0) = (w1 * w1) - 1;
        PP_t_1(1,1) = (h1 * h1) - 1;
        PP_t_1 *= (w1 * h1 / 12);
        
        Mat_<float> PP_t_2(3,3);
        PP_t_2.setTo(0.0f);
        PP_t_2(0,0) = (w2 * w2) - 1;
        PP_t_2(1,1) = (h2 * h2) - 1;
        PP_t_2 *= (w2 * h2 / 12);
        
        Mat_<float> pp_t_1 = (Mat_<float>(3,3) << 
                                ((w1-1)*(w1-1)), (w1-1)*(h1-1), 2*(w1-1), 
                                (w1-1)*(h1-1), (h1-1)*(h1-1), 2*(h1-1), 
                                2*(w1-1), 2*(h1-1), 4);
        pp_t_1 *= 0.25;
        
        Mat_<float> pp_t_2 = (Mat_<float>(3,3) << 
                                ((w2-1)*(w2-1)), (w2-1)*(h2-1), 2*(w2-1), 
                                (w2-1)*(h2-1), (h2-1)*(h2-1), 2*(h2-1), 
                                2*(w2-1), 2*(h2-1), 4);
        pp_t_2 *= 0.25;
        
        Mat A1 = e1_x.t() * PP_t_1 * e1_x;
        Mat B1 = e1_x.t() * pp_t_1 * e1_x;
        
        Mat A2 = F_t * PP_t_2 * F;
        Mat B2 = F_t * pp_t_2 * F;
        
        Mat z1, z2;
        
        // take only the relevant 2 x 2 matrix
        Mat chol1 = A1.rowRange(0,2).colRange(0,2).clone();
        Mat chol2 = A2.rowRange(0,2).colRange(0,2).clone();
        
        if (Cholesky(chol1.ptr<float>(), chol1.step, chol1.cols, 0, 0, 0)){
                Mat D = chol1; // upper triangular
                std::cout << "A1: " << A1 << std::endl;
                std::cout << "D1: " << D << std::endl;
                Mat D_t = chol1.t(); // lower triangular
                Mat D_inv = D.inv();
                Mat E, V;
                Mat B = B1.rowRange(0,2).colRange(0,2);
                eigen(D_inv.t() * B * D_inv, E, V);
                Mat y = V.row(0).clone();
                y = y.t();
                z1 = D_inv * y;
        }else{
                throw runtime_error("Could not decompose A1");
        }
        
        if (Cholesky(chol2.ptr<float>(), chol2.step, chol2.cols, 0, 0, 0)){
                Mat D = chol2; // upper triangular
                std::cout << "A2: " << A2 << std::endl;
                std::cout << "D2: " << D << std::endl;
                Mat D_t = chol2.t(); // lower triangular
                Mat D_inv = D.inv();
                Mat E, V;
                Mat B = B2.rowRange(0,2).colRange(0,2);
                eigen(D_inv.t() * B * D_inv, E, V);
                Mat y = V.row(0).clone(); // eigenvector with largest eigenvalue
                y = y.t();
                z2 = D_inv * y;
        }else{
                throw runtime_error("Could not decompose A2");
        }
        
        std::cout << "z1: " << z1 << std::endl;
        std::cout << "z2: " << z2 << std::endl;
        normalize(z1, z1);
        normalize(z2, z2);
        std::cout << "normalized z1: " << z1 << std::endl;
        std::cout << "normalized z2: " << z2 << std::endl;
        
        Mat tmp = 0.5 * (z1 + z2);
        Mat_<float> z(3,1);
        z.setTo(0.0f);
        z(0,0) = tmp.at<float>(0,0);
        z(1,0) = tmp.at<float>(1,0);
        std::cout << "z: " << z << std::endl;
        
        Mat wt1 = e1_x * z;
        wt1 /= wt1.at<float>(2,0);
        std::cout << "wt1: " << wt1 << std::endl;
        Mat wt2 = F * z;
        wt2 /= wt2.at<float>(2,0);
        std::cout << "wt2: " << wt2 << std::endl;
        
        // The projective transform
        Mat_<float> H_p_1 = Mat::eye(3,3,CV_32F);
        H_p_1(2,0) = wt1.at<float>(0,0);
        H_p_1(2,1) = wt1.at<float>(1,0);
        
        Mat_<float> H_p_2 = Mat::eye(3,3,CV_32F);
        H_p_2(2,0) = wt2.at<float>(0,0);
        H_p_2(2,1) = wt2.at<float>(1,0);
        
        // The similarity transform
        Mat_<float> H_r_1 = Mat::eye(3,3,CV_32F);
        H_r_1(0,0) = F.at<float>(2,1) - wt1.at<float>(1,0) * F.at<float>(2,2);
        H_r_1(0,1) = wt1.at<float>(0,0) * F.at<float>(2,2) - F.at<float>(2,0);
        H_r_1(1,0) = -H_r_1.at<float>(0,1);
        H_r_1(1,1) = H_r_1.at<float>(0,0);
        H_r_1(1,2) = F.at<float>(2,2);
        
        Mat_<float> H_r_2 = Mat::eye(3,3,CV_32F);
        H_r_2(0,0) = wt2.at<float>(1,0) * F.at<float>(2,2) - F.at<float>(1,2);
        H_r_2(0,1) = F.at<float>(0,2) - wt2.at<float>(0,0) * F.at<float>(2,2);
        H_r_2(1,0) = -H_r_2.at<float>(0,1);
        H_r_2(1,1) = H_r_2.at<float>(0,0);
        H_r_2(1,2) = 0.0f;
        
        // Store
        this->Hp1 = H_p_1;
        this->Hp2 = H_p_2;
        
        // convert to 2x3 affine transformation and store
        this->Hr1 = H_r_1.rowRange(0,2).clone();
        this->Hr2 = H_r_2.rowRange(0,2).clone();
}

int cEpipolar::applyHomography(){
        float min_v = std::numeric_limits<float>::max();
        for(int y = 0; y < this->img1.rows; y++){
                for(int x = 0; x < this->img1.cols; x++){
                        Point2f tmp(x,y);
                        Mat_<float> p = this->getHomogeneous(tmp);
                        Mat_<float> p_trans1 = this->Hr1 * this->Hp1 * p;
                        Mat_<float> p_trans2 = this->Hr2 * this->Hp2 * p;
                        p_trans1 /= p_trans1(2,0);
                        p_trans2 /= p_trans2(2,0);
                        if(p_trans1(1,0) < min_v){
                                min_v = p_trans1(1,0);
                        }
                        if(p_trans2(1,0) < min_v){
                                min_v = p_trans2(1,0);
                        }
                }
        }
        // Update affine homography
        this->Hr1.at<float>(1,2) += min_v;
        this->Hr2.at<float>(1,2) = min_v;
                
        Mat i1_p, i2_p;
        warpPerspective(this->img1, i1_p, this->Hp1, this->img1.size());
        warpPerspective(this->img2, i2_p, this->Hp2, this->img2.size());
        
        imshow("left_image",i1_p);
        imshow("right_image",i2_p);
        waitKey(0);
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
//     cout<<endl<<endl<<"Q3....."<<endl;
//     cDisparity disparity;
//     disparity.loadImages();
//     disparity.computeDisparity();

    // Q4: Rectifying image pair
    cout<<endl<<endl<<"Q4....."<<endl;
    epipolar.rectify();
    epipolar.applyHomography();

    return 0;
}
