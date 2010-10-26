#include <iostream>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/Core>
// do the dense case first

using namespace std;
using namespace Eigen;

static clock_t tic_timestart;
void tic(void) {
  tic_timestart = clock();
}

double toc(void) {
  clock_t tic_timestop;
  tic_timestop = clock();
  printf("time: %8.2f.\n", (double)(tic_timestop - tic_timestart) / CLOCKS_PER_SEC);
  return (double)(tic_timestop - tic_timestart) / CLOCKS_PER_SEC;
}

double tocq(void) {
    clock_t tic_timestop;
    tic_timestop = clock();
    return (double)(tic_timestop - tic_timestart) / CLOCKS_PER_SEC;
}

VectorXd lsmr(MatrixXd & A, VectorXd & b);

int main(int argc, char** argv)
{
  if (argc != 4) {
    cout << "not enough input args..." << endl;
    return -1;
  }
  
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int num_loops = atoi(argv[3]);
  
  cout << m << ", " << n << endl;
  
  MatrixXd A = MatrixXd::Random(m,n);
  VectorXd b = VectorXd::Random(m);
  
  VectorXd x1;
  //for(int i = 0; i < 100000; i++) {
  cout << "Householder QR" << endl;
  tic();
  for(int i = 0 ; i < num_loops; i ++)
    x1 = A.householderQr().solve(b);
  toc();
  
  //}
  //cout << x << endl << endl;
  // run lsmr...
  VectorXd x2;
  cout << endl << "LSMR" << endl;
  tic();
  for(int i = 0 ; i < num_loops; i ++)
    x2 = lsmr(A,b);
  toc();

  cout << "error: " << (x1-x2).norm() << endl;
}

VectorXd lsmr(MatrixXd & A, VectorXd & b)
{
  double a_tol = 1e-6; double b_tol = 1e-6;
  double lambda = 0; double conlim = 1e+8;
  
  VectorXd u = b;
  double beta = u.norm();
  u.normalize();
  
  VectorXd v = A.transpose()*u;
  double alpha = v.norm();
  v.normalize();
  
  int m = A.rows(); int n = A.cols();
  int minDim = min(m, n);
  int maxiters = minDim;

  // skip local ortho...
  
  // vars for first iter
  int iter = 0;
  double zetabar = beta*alpha;
  double alphabar = alpha;
  double rho = 1.0;
  double rhobar = 1.0;
  double cbar = 1.0;
  double sbar = 0.0;
  
  VectorXd h = v;
  VectorXd hbar = VectorXd::Zero(n);
  VectorXd x = VectorXd::Zero(n);
  
  // vars for estimating ||r||
  double betadd = beta;
  double betad = 0.0;
  double rhodold = 1.0;
  double tautildeold = 0.0;
  double thetatilde = 0.0;
  double zeta = 0.0;
  double d = 0.0;
  
  // init vars for estimating ||A|| and cond(A)
  double normA2 = alpha*alpha;
  double maxrbar = 0.0;
  double minrbar = 1e100;
  
  // stopping crit
  double normb = beta;
  //double istop = 0;
  double ctol = 0;
  if (conlim > 0)
    ctol = 1.0/conlim;
  double normr = beta;
  
  double normAr = alpha*beta;
  if (normAr == 0) {
    cout << "exact solution is 0" << endl;
    return x;
  }
  
  while (iter < maxiters) {  // loop count = 100
    u = A*v - alpha*u;
    beta = u.norm();
    if (beta > 0) {
      u.normalize();
      v = A.transpose()*u - beta*v;
      alpha = v.norm();
      if (alpha > 0) 
        v.normalize();
    }
    
    // construct rotation Qhat
    double alphahat = sqrt(alphabar*alphabar + lambda*lambda);  // no regularization term
    double chat = alphabar/alphahat;
    double shat = lambda/alphahat;
    
    // plane rotations...
    
    double rhoold = rho;
    rho = sqrt(alphahat*alphahat + beta*beta);
    double c = alphahat/rho;
    double s = beta/rho;
    double thetanew = s*alpha;
    alphabar = c*alpha;
        
    double rhobarold = rhobar;
    double zetaold = zeta;
    double thetabar = sbar*rho;
    double rhotemp = cbar*rho;
    rhobar = sqrt( cbar*rho*cbar*rho + thetanew*thetanew );
    cbar *= rho/rhobar;
    sbar = thetanew/rhobar;
    zeta = cbar*zetabar;
    zetabar = -sbar*zetabar;
    
    // update h, h_hat, x
    //cout << thetabar << ", " << rho << ", " << rhoold << ", " << rhobarold << endl;
    hbar = h - (thetabar*rho/(rhoold*rhobarold))*hbar;
    //cout << hbar << endl;
    //cout << zeta << ", " << rho << ", " << rhobar << endl;
    x += (zeta/(rho*rhobar))*hbar;
    //cout << x << endl;
    h = v - (thetanew/rho)*h;
    
    // estimate of ||r||
    double betaacute = chat*betadd;
    double betacheck = -shat*betadd;
    
    double betahat = c*betaacute;
    betadd = -s*betaacute;
    
    double thetatildeold = thetatilde;
    double rhotildeold = sqrt( rhodold*rhodold + thetabar*thetabar );
    double ctildeold = rhodold/rhotildeold;
    double stildeold = thetabar/rhotildeold;
    thetatilde = stildeold*rhobar;
    rhodold = ctildeold*rhobar;
    betad = -stildeold*betad + ctildeold*betahat;
    
    tautildeold = (zetaold - thetatildeold*tautildeold)/rhotildeold;
    double taud = (zeta - thetatilde*tautildeold)/rhodold;
    d = d + betacheck*betacheck;
    normr = sqrt(d + (betad - taud)*(betad - taud) + betadd*betadd);
    
    // estimate ||A||
    normA2 += beta*beta;
    double normA = sqrt(normA2);
    normA2 += alpha*alpha;
    
    maxrbar = max(maxrbar, rhobarold);
    if (iter > 1)
      minrbar = min(minrbar, rhobarold);
    double condA = max(maxrbar, rhotemp)/min(minrbar, rhotemp);
    
    // stopping crtierion
    normAr = abs(zetabar);
    double normx = x.norm();
    
    double test1 = normr/normb;
    double test2 = normAr / (normA*normr);
    double test3 = 1.0/condA;
    double t1 = test1 / (1 + normA*normx/normb);
    double rtol = b_tol + a_tol*normA*normx/normb;
    
    // skip error checking
    
    // check tests
    if (test3 <= ctol || test2 <= a_tol || test1 <= rtol)
      break;
    
    //printf("%d\t%f\t%0.3f\t%0.3f\t%f\t%0.1f\n", iter, x(0), normr, normAr, test1,test2);
    
    iter++;
  }
  
  //cout << x << endl;
  return x;
}