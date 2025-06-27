
data {
    real y_obs;   // Observed data point
}

parameters {
    real mu;             // Mean parameter
    real<lower=0> sigma; // Standard deviation (must be positive)
}

model {
    // Priors
    mu ~ normal(0, 5);         // Prior: mu ~ N(0,5)
    sigma ~ normal(0, 1);      // Half-Normal prior for sigma

    // Likelihood
    y_obs ~ normal(mu, sigma);
}
