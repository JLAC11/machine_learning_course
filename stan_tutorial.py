import stan
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model_str = """
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
   rate_a ~ beta(3,25);
   rate_b ~ beta(3,25);
   success_a ~ binomial(sample_a,rate_a);
   success_b ~ binomial(sample_b,rate_b);
   
}

generated quantities {
    real rate_diff;
    rate_diff = rate_b-rate_a; // Si es mayor que 0, entonces B tiene mayor tasa
}
"""
data_dict = {"sample_a": 16, "sample_b": 16, "success_a": 6, "success_b": 10}

posterior = stan.build(model_str, data=data_dict)
df = posterior.sample().to_frame()
print(df)
sns.jointplot(
    data=df, x="rate_a", y="rate_b", kind="kde", joint_kws={"fill": True}
)
plt.show()

# Expected profit functions:
# Only brochure: - 30 + 1000p_a
# Brochure and fish: -330 + 1000p_b

profit_a = 1000 * df.rate_a - 30
profit_b = 1000 * df.rate_b - 330

profit_a.plot.kde()
profit_b.plot.kde()
plt.legend(("Only brochure", "Brochure and fish"))
plt.xlabel("Expected value ($)")
plt.ylabel("Density")
plt.show()
