data {
   int sample_a;
   int sample_b;

   int success_a;
   int success_b;
}

parameters {
   real<lower=0,upper=1> rate_a;
   real<lower=0,upper=1> rate_b;
}

model {
   rate_a ~ uniform(0,1);
   rate_b ~ uniform(0,1);

   success_a ~ binomial(sample_a,rate_a);
   success_b ~ binomial(sample_b,rate_b);
   
}

generated quantities {
    real rate_diff;
    rate_diff <- rate_b-rate_a # Si es mayor que 0, entonces B tiene mayor tasa
}