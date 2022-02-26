data {
int<lower=1> N_y;
int<lower=1> p;
matrix[N_y, p] X;
vector[N_y] y;
int<lower=0> N_new;            
matrix[N_new, p] X_new;      
}

parameters {
real alpha;
vector[p] beta;
real<lower=0> sigma_y;
}

model {
alpha ~ normal(0,1);
beta ~ normal(0, 1);
sigma_y ~ exponential(1);
y ~ normal(alpha + X * beta, sigma_y);
}

generated quantities {
  vector[N_new] mu_pred;
  vector[N_new] y_pred;

  for (n in 1:N_new) {

    mu_pred[n] = alpha + X_new[n] * beta;
    y_pred[n] = normal_rng(mu_pred[n], sigma_y);

   }
}

