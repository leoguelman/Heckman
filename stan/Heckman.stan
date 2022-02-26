data {
// dimensions
int<lower=1> N;                 // total number of obs
int<lower=1, upper=N> N_y;      // number of y obs (excluding NAs)
int<lower=1> p;                 // number of covariates for the outcome equation (excluding intercept)
int<lower=1> q;                 // number of covariates for the missing equation (excluding intercept)
// covariates  
matrix[N_y, p] X;               // covariates for the outcome equation (excluding NAs in y)
matrix[N, q] Z;                 // covariates for the missing equation 
// responses
int<lower=0, upper=1> D[N];     // Missing value indicator
vector[N_y] y;                  // Outcome (excluding NAs)
int<lower=0> N_new;             // number of scoring obs
matrix[N_new, p] X_new;         // scoring covariate matrix
}

parameters {
real alpha;
real delta;
vector[p] beta;
vector[q] gamma;
real<lower=-1,upper=1> rho_yd;
real<lower=0> sigma_y;
}

model {

alpha ~ normal(0,1);
delta ~ normal(0,1);
beta ~ normal(0, 1);
gamma ~ normal(0, 1);
rho_yd ~ uniform(-1, 1);
sigma_y ~ exponential(1);
{
// log-likelihood
vector[N_y] mu_y;
vector[N] mu_D;
int ny;

mu_y = alpha + X * beta;
mu_D = delta + Z * gamma;

ny = 1;
for(n in 1:N) {
if(D[n] > 0) {
target += normal_lpdf(y[ny] | mu_y[ny], sigma_y) + log(Phi((mu_D[n] + rho_yd / sigma_y * (y[ny] - mu_y[ny])) / sqrt(1 - rho_yd^2)));
ny += 1;
}
else {
target += log(Phi(-mu_D[n]));
}
}
}
}

generated quantities {
  vector[N_new] mu_pred;
  vector[N_new] y_pred;

  for (n in 1:N_new) {

    mu_pred[n] = alpha + X_new[n] * beta;
    y_pred[n] = normal_rng(mu_pred[n], sigma_y);

   }
}

