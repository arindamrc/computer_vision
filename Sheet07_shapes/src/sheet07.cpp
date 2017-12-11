#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <math.h>

using namespace std;
using namespace cv;

#define UNIT_VARIANCE 10.0
#define MIN_ERR 10e-5
#define MAX_ITER 1000

string file_for_procustes = "./images/hands_orig_train.txt";
string train_file_shape = "./images/hands_aligned_train.txt";
string test_file_shape = "./images/hands_aligned_test.txt";

/////////////////////////////////////////////////
//////// class declaration for task 1 ///////////
/////////////////////////////////////////////////
class ProcrustesAnalysis{
public:
    ProcrustesAnalysis(int maxIter, float minErr);
    ~ProcrustesAnalysis(){};
    bool LoadData(string in_fpath);
    void AlignData();

private:
    Mat findTransformation(Mat_<float>& A, Mat& meanShape);
    void displayShape(Mat& in_shapes, string header, int waitFlag, float t);
    void preProcessShapes(int cidx);
    void normalize_shape(Mat_<float>& s);
    void recenter_shape(Mat_<float>& s);
    void copyData();
    Mat data;
    vector<Mat_<float>> shapes;
    int m_iter, m_maxIter;
    float m_err, m_minErr;
    int num_coords, num_sampls;
};


ProcrustesAnalysis::ProcrustesAnalysis(int maxIter, float minErr)
{
        this->m_maxIter = maxIter;
        this->m_minErr = minErr;
        m_iter = 0;
        m_err = 0.0f;
}

bool ProcrustesAnalysis::LoadData(string in_fpath){
        ifstream in(in_fpath.c_str());
        if(!in) return false;
        in >> num_coords;
        in >> num_sampls;
        Mat indata(num_coords,num_sampls,CV_32F);
        indata.setTo(0);
        for(int row=0; row<indata.rows; ++row){
                for(int col=0; col<indata.cols; ++col){
                        in >> indata.at<float>(row,col);
                }
        }
        data = indata;
        shapes.resize(data.cols);
        return true;
}

void ProcrustesAnalysis::displayShape(Mat& in_shapes, string header, int waitFlag, float t){
    // init interm. parameters
    int scl=500;
    double maxval;
    RNG rng;
    Mat shapes_dup = in_shapes.clone();
    minMaxLoc(shapes_dup,NULL,&maxval,NULL,NULL);
    shapes_dup *= (scl*0.8/maxval);

    Mat dispImg(scl,scl,CV_8UC3);
    Scalar color(0,0,0);
    dispImg.setTo(color);
    int lstx=shapes_dup.rows/2-1;

    if(0==waitFlag){
        color[0]=20; color[1]=40; color[2]=180;
    }

    // draw each input shape in a different color
    for(int cidx=0; cidx<shapes_dup.cols; ++cidx){
        if(1==waitFlag){
            color[0]=rng.uniform(0,256); color[1]=rng.uniform(0,256); color[2]=rng.uniform(0,256);
        }
        for(int ridx=0; ridx<lstx-1; ++ridx){
            Point2i startPt(shapes_dup.at<float>(ridx,cidx) + t,shapes_dup.at<float>(ridx+lstx+1,cidx) + t);
            Point2i endPt(shapes_dup.at<float>(ridx+1,cidx) + t,shapes_dup.at<float>(ridx+lstx+2,cidx) + t);
            line(dispImg,startPt,endPt,color,2);
        }
        imshow(header.c_str(),dispImg);
        waitKey(20);
    }
    if(1==waitFlag){
        cout<<"press any key to continue..."<<endl;
        waitKey(0);
    }
}

// split the single column shape vector into 2 col one. col 1 holds x 
// coords and col 2 holds corresponding y coords.
void ProcrustesAnalysis::preProcessShapes(int sidx)
{
        int mid = data.rows / 2;
        cv::Mat_<float> shape(mid, 2, 0.0f);
        Mat s = data.col(sidx);
        s.rowRange(0,mid).copyTo(shape.col(0)); // copy x
        s.rowRange(mid,s.rows).copyTo(shape.col(1)); // copy y
        shapes[sidx] = shape;
}


// s is a 2 column matrix with all x coords in col 1 and y coords in col 2. Each row represents a landmark point.
// Using the L2 norm
void ProcrustesAnalysis::normalize_shape(Mat_<float>& s)
{
        float normalizer = norm(s);
        for(int r = 0; r < s.rows; r++){
                s.at<float>(r,0) /= normalizer;
                s.at<float>(r,1) /= normalizer;
        }
}

// s is a single column matrix with all x coords followed by y coords.
// Find the center of shape and subtract it from all landmark points
// Center is simply the x and y coordinate averages calculated independently
void ProcrustesAnalysis::recenter_shape(Mat_<float>& s){
        double x_avg = mean(s.col(0))(0);
        double y_avg = mean(s.col(1))(0);
        for(int r = 0; r < s.rows; r++){
                s.at<float>(r,0) -= x_avg;
                s.at<float>(r,1) -= y_avg;
        }
}

// Perform generalized procrustes analysis to align data
void ProcrustesAnalysis::AlignData()
{
        Mat_<float> err; // shape error
        double avgErr = 0.0; // average error
        
        // Reshape the shape matrix into a vector of N x 2 shapes.
        for(int sidx = 0; sidx < data.cols; sidx++){
                preProcessShapes(sidx);
        }
        
        // Mean shape holder
        Mat_<float> sShape(shapes[0].rows, shapes[0].cols, 0.0);
        
        // Display the initial data
        this->displayShape(data, "initial", 1, 0.0);

        
        Mat_<float> mu = shapes[0]; // Initialize the mean to the first shape: GPA
        
        // recenter and normalize mean
        recenter_shape(mu); 
        normalize_shape(mu);
        
        for(this->m_iter = 0; this->m_iter < this->m_maxIter; this->m_iter++){ // for many iterations
                for(int sidx = 0; sidx < shapes.size(); sidx++){ // for every shape
                        Mat_<float> shape = shapes[sidx];
                        
                        // recenter and normalize shape
                        recenter_shape(shape);
                        normalize_shape(shape);
                        
                        // find the transformation
                        Mat T = this->findTransformation(shape, mu);
                        
                        // apply the transformation
                        Mat_<float> nShape = shape * T;
                        
                        // find the error between the mean and the transformed shape
                        err = mu - nShape;
                        
                        // add up the new shapes
                        sShape += nShape;
                        
                        // add up the error
                        avgErr += norm(err);
                        
                        shapes[sidx] = nShape;
                }
                
                avgErr = avgErr / shapes.size(); // find average error
                mu = (sShape / shapes.size()); // find new mean shape
                
                if(avgErr <= this->m_minErr){ 
                        // break if error falls beneath a certain threshold
                        break;
                }

                if(m_iter % 200 == 0){
                        std::cout << "Avg err: " << avgErr << std::endl;
                        std::string header = "shape ";
                        header += std::to_string(m_iter);
                        this->copyData();
                        this->displayShape(data, header, 1, 250.0);
                }
                avgErr = 0.0;
                sShape.setTo(0.0f);
        }
        
}

// Find the transformation between the given shape and the mean
// This is found by solving Least Squares by SVD.
Mat ProcrustesAnalysis::findTransformation(Mat_<float>& shape, Mat& mu)
{
        Mat T;
        T.setTo(0.0f);
        Mat w,u,vt;
        
        // for details look at ICP or point cloud transformations using SVD.
        SVDecomp(mu.t() * shape, w, u, vt);
        T = vt.t() * u.t();
        return T;
}

// copy the data from the shape vector to data matrix. This aids in display.
void ProcrustesAnalysis::copyData()
{
        int mid = data.rows / 2;
        for(int sidx = 0; sidx < shapes.size(); sidx++){
                shapes[sidx].col(0).copyTo(data.col(sidx).rowRange(0,mid));
                shapes[sidx].col(1).copyTo(data.col(sidx).rowRange(mid,data.rows));
        }
}



/////////////////////////////////////////////////
////  class declaration for tasks 2 and 3  //////
/////////////////////////////////////////////////
class ShapeModel{
public:
        ShapeModel(){rng(10); scl=400;};
        ~ShapeModel(){};
        void loadData(const string& fileLoc, Mat& data);
        void trainModel(int K);
        void inference();
        /* utilities */
        void displayShape(Mat& shapes, string header, int waitFlag=1, float t=0.0f);
        void displayModel();
        /* variables */
        Mat trainD;
        Mat testD;

private:
        Mat meanShape;
        Mat prinComp;
        Mat prinVal;
        RNG rng;
        int scl;
        void center(Mat& s);
        void normalize(Mat& s);
        Mat findTransformation(Mat& shape, Mat& mu);
};

void ShapeModel::center(cv::Mat& s)
{
        int mid = s.rows/2;
        float x_avg = mean(s.rowRange(0,mid))(0);
        float y_avg = mean(s.rowRange(mid,s.rows))(0);
        for(int ridx = 0; ridx < mid; ridx ++){
                s.at<float>(ridx,0) -= x_avg;
                s.at<float>(ridx+mid,0) -= y_avg;
        }
}

void ShapeModel::normalize(cv::Mat& s)
{
        float norm_l2 = norm(s);
        for(int ridx = 0; ridx < s.rows; ridx++){
                s.at<float>(ridx,0) /= norm_l2;
        }
}


void ShapeModel::loadData(const string& fileLoc, Mat& data){

    // check if file exists
    ifstream iStream(fileLoc.c_str());
    if(!iStream){
        cout<<"file for load data cannot be found"<<endl;
        exit(-1);
    }

    // read aligned hand shapes
    int rows, cols;
    iStream >> rows;
    iStream >> cols;
    data.create(rows,cols,CV_32F);
    data.setTo(0);
    float *dptr;
    for(int ridx=0; ridx<data.rows; ++ridx){
        dptr = data.ptr<float>(ridx);
        for(int cidx=0; cidx<data.cols; ++cidx, ++dptr){
            iStream >> *dptr;
        }
    }
    iStream.close();
}

void ShapeModel::displayShape(Mat& in_shapes, string header, int waitFlag, float t){
        // init interm. parameters
        Mat shapes = in_shapes.clone();
        double maxval;
        minMaxLoc(shapes,NULL,&maxval,NULL,NULL);
        shapes *= (scl*0.5/maxval);
        
        Mat dispImg(scl,scl,CV_8UC3);
        Scalar color(0,0,0);
        dispImg.setTo(color);
        int lstx=shapes.rows/2-1;

        if(0==waitFlag){
                color[0]=20; color[1]=40; color[2]=180;
        }

        // draw each input shape in a different color
        for(int cidx=0; cidx<shapes.cols; ++cidx){
                if(1==waitFlag){
                        color[0]=rng.uniform(0,256); color[1]=rng.uniform(0,256); color[2]=rng.uniform(0,256);
                }
                for(int ridx=0; ridx<lstx-1; ++ridx){
                        Point2i startPt(shapes.at<float>(ridx,cidx) + t,shapes.at<float>(ridx+lstx+1,cidx) + t);
                        Point2i endPt(shapes.at<float>(ridx+1,cidx) + t,shapes.at<float>(ridx+lstx+2,cidx) + t);
                        line(dispImg,startPt,endPt,color,2);
                }
                imshow(header.c_str(),dispImg);
                waitKey(10);
        }
        if(1==waitFlag){
        cout<<"press any key to continue..."<<endl;
        waitKey(0);
        }
}

void ShapeModel::displayModel()
{
        Mat phi = prinComp.clone();
        Mat mu = meanShape.clone();
        int K = phi.cols; // The number of orthogonal bases
        Mat h = Mat::zeros(K, 1, phi.type());
        
        // show only the mean shape first
        this->displayShape(mu, "mean", 1, this->scl/2.0);
        
        Mat shapes(mu.rows, 1, mu.type());
        
        for(int k = 0; k < K; k++){
                float val = -0.3f;
                while(val <= 0.3f){
                        if(fabs(val) < 0.001f){
                                val += 0.1;
                                continue;
                        }
                        h.at<float>(k,0) = val;
                        Mat wi = mu + (phi * h);
                        hconcat(shapes, wi, shapes);
                        this->displayShape(wi, "means varied", 1, this->scl/2.0);
                        val += 0.1;
                }
                h.at<float>(k,0) = 0.0f;
        }
}


// Find the transformation from shape x to Y
// This is found by solving Least Squares by SVD.
Mat ShapeModel::findTransformation(Mat& x, Mat& Y)
{
        int mid = x.rows/2;
        Mat x_(mid, 2, x.type());
        Mat Y_(mid, 2, Y.type());
        x.rowRange(0,mid).copyTo(x_.col(0));
        x.rowRange(mid,x.rows).copyTo(x_.col(1));
        Y.rowRange(0,mid).copyTo(Y_.col(0));
        Y.rowRange(mid,x.rows).copyTo(Y_.col(1));
        
        Mat T;
        T.setTo(0.0f);
        Mat w,u,vt;
        
        // for details look at ICP or point cloud transformations using SVD.
        SVDecomp(Y_.t() * x_, w, u, vt);
        T = vt.t() * u.t();
        return T;
}

void ShapeModel::inference()
{
        // center the test data
        Mat Y = this->testD.clone();
        int mid = this->testD.rows / 2;
        
        // recenter and normalize test shape
        center(Y);
        normalize(Y);
        
        Mat x_bar = meanShape.clone();
        Mat P = prinComp.clone();
        
        Mat b = Mat::zeros(P.cols, 1, P.type()); // initialize b to 0
        
        Mat x;
        
        int count = 0;
        
        while(true){
                x = x_bar + (P * b);
                
                // Find transformation.
                // This method is outlined in "An Introduction to Active Shape Models" by Tim Cootes
                // Though the transformation can also be found by using SVD. 
                // Simply invoke this->findTransformation
                float x_norm2 = pow(norm(x),2.0);
                float m = x.dot(Y) / x_norm2;
                float n = 0.0f;
                for(int i = 0; i < mid; i++){
                        n += (x.at<float>(i,0) * Y.at<float>(i+mid,0) - x.at<float>(i+mid,0) * Y.at<float>(i,0));
                }
                n /= x_norm2;
                float s = pow(m*m + n*n, 0.5);
                float theta = atan2(n,m);
                Mat_<float> rot(2,2);
                float alpha = s*cos(theta), beta = s*sin(theta);
                rot(0,0) = alpha;
                rot(0,1) = -beta;
                rot(1,0) = beta;
                rot(1,1) = alpha;
                
                Mat rot_inv = rot.inv();
                Mat y(Y.rows, Y.cols, Y.type());
                
                // Apply the inverse transformation
                for(int i = 0; i < mid; i++){
                        Mat_<float> pt(2,1);
                        pt(0,0) = Y.at<float>(i,0);
                        pt(1,0) = Y.at<float>(i+mid,0);
                        pt = rot_inv * pt;
                        y.at<float>(i,0) = pt(0,0);
                        y.at<float>(i+mid,0) = pt(1,0);
                }
                
                y = y / (y.dot(x_bar));
                
                // Strictly speaking only the following is enough. 
                // Here the test shape is roughly in the same coordinate system as the training data.
                Mat b_new = P.t() * (y - x_bar);
                
                float err = norm(b - b_new);
                
                if(err < MIN_ERR || count >= MAX_ITER){
                        // Stop when there is no appreciable update or when the maximum iterations are done
                        break;
                }
                
                count++;
                b = b_new;
        }
        
        // The final inference results
        x = x_bar + (P * b);
        hconcat(Y, x, Y);
        this->displayShape(Y, "original+inferred", 1, this->scl/2.0);
}

void ShapeModel::trainModel(int K)
{
        // Find mean shape
        int D = this->trainD.rows; // Dimensions are the number of landmark point per shape
        
        // recenter and normalize the training shapes
        for(int sidx = 0; sidx < this->trainD.cols; sidx++){
                Mat c = this->trainD.col(sidx);
                center(c);
                normalize(c);
                // show the original shapes
        }
        
        
        // Find the mean shape
        Mat mu;
        reduce(trainD, mu, 1, CV_REDUCE_AVG, -1);
        
        // subtract mean shape from the training data
        Mat W(trainD.rows, trainD.cols, trainD.type());
        for(int cidx = 0; cidx < trainD.cols; cidx++){
                Mat diff = trainD.col(cidx) - mu;
                diff.copyTo(W.col(cidx));
        }
        
        
        // Decompose using SVD
        Mat L2_vec, U, Ut;
        SVDecomp(W * W.t(), L2_vec, U, Ut);
        
        // find K
        float E_total = cv::sum(L2_vec)(0);
        float E_90 = 0.9 * E_total;
        float E_sum = 0.0f;
        int k = 0;
        for(k = 0; k < L2_vec.rows; k++){
                E_sum += L2_vec.at<float>(k,0);
                if(E_sum >= E_90){
                        break;
                }
        }
        K = k+1; // row indices are 0 based
        
        // convert L2_vec to diagonal matrix
        Mat L2 = Mat::zeros(D,D,W.type());
        L2_vec.copyTo(L2.diag(0));
        
        // Estimate noise
        float sigma2 = 0.0;
        for(int j = K+1; j < D; j++){
                sigma2 += L2.at<float>(j,j);
        }
        sigma2 /= (D-K);
        
        // Find principal components with removed noise
        Mat U_K = U.colRange(0,K); // First K cols of U (eigenvectors)
        Mat L2_K = L2.colRange(0,K).rowRange(0,K); // First K rows and K cols of L2
        Mat I = Mat::eye(L2_K.rows, L2_K.cols, L2_K.type());
        
        Mat sqroot;
        cv::sqrt(L2_K - sigma2*I, sqroot);
        
//         sqroot.setTo(0.0f); Uncomment this to keep the noise in the data!
//         sqroot.diag(0).setTo(1.0f); // Setting sqroot to identity keeps U_K unaltered.
        
        Mat phi = U_K * sqroot;
        
        this->prinComp = phi;
        
        this->meanShape = mu;
        
}



int main(){

//    // Procrustes Analysis
//    ProcrustesAnalysis proc(MAX_ITER, MIN_ERR);
//    proc.LoadData(file_for_procustes);
//    proc.AlignData();
   
    // Shape Analysis
    // training procedure
        ShapeModel model;
        model.loadData(train_file_shape,model.trainD);
        int K = 30;
        model.displayShape(model.trainD,string("trainingShapes"), 1, 0.0f);
        model.trainModel(K);
        model.displayModel();

        // testing procedure
        model.loadData(test_file_shape,model.testD);
        model.inference();

        cout <<                                                                                                   endl;
        cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
        cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
        cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << endl;
        cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
        cout << "////////////////////////////////////////////////////////////////////////////////////////////" << endl;
        cout <<                                                                                                   endl;

        cout << "exiting code" << endl;
        return 0;
}
