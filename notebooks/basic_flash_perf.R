# --- Setup ---
set.seed(1)

# Packages
if (!requireNamespace("flashier", quietly = TRUE)) install.packages("flashier")
if (!requireNamespace("ebnm", quietly = TRUE)) install.packages("ebnm")
library(flashier)
library(ebnm)

# --- Config ---
n <- 50
p <- 40
noise_sd <- 0.1
n_iter <- 100

rmse <- numeric(n_iter)
rmse2 <- numeric(n_iter)
rmse3 <- numeric(n_iter)


# --- Simulation + fit loop ---
for (i in seq_len(n_iter)) {
  set.seed(i)
  # Rank-1 truth
  u <- runif(n)
  v <- runif(p)
  S <- outer(u, v)                          # signal (n x p)

  # Noise + observed matrix
  Z <- S + matrix(rnorm(n * p, sd = noise_sd), n, p)

  # Fit flashier with GB (for L) and point-Laplace (for F)
  # You can swap which side gets which prior by swapping the two entries.
  fit <- flash(
    data = Z,
    greedy_Kmax = 10,
    ebnm_fn = ebnm_ash,
    backfit = TRUE,
    verbose = 0
  )

  fit2 <- flash(
    data = Z,
    greedy_Kmax = 10,
    ebnm_fn = ebnm_point_laplace,
    backfit = TRUE,
    verbose = 0)
  fit3 <- flash(
    data = Z,
    greedy_Kmax = 10,
    ebnm_fn = ebnm_point_exponential,
    backfit = TRUE,
    verbose = 0)
  Zhat <- fitted(fit)
  Zhat2 <- fitted(fit2)
  Zhat3 <- fitted(fit3)
  # n x p
  rmse[i] <- sqrt(mean((Zhat - S)^2))
  rmse2[i] <- sqrt(mean((Zhat2 - S)^2))
  rmse3[i] <- sqrt(mean((Zhat3 - S)^2))
  print(i)
}


quantile(rmse)

df= data.frame( RMSE= c(rmse,
                        rmse2,
                        rmse3),
                method= factor(rep( c("ash",
                                      "pt Laplace",
                                      "pt exp"), each=n_iter)
                               )
                )
library(ggplot2)

ggplot(df, aes(x=RMSE, fill=method))+
  geom_density(alpha=0.5)
library(dplyr)
# --- Summary ---
df%>%
  group_by(method)%>%
  summarize( mean= mean(RMSE),
             sd=sd(RMSE))



#method       mean      sd
#<fct>       <dbl>   <dbl>
#  1 ash        0.0214 0.00145
#2 pt exp     0.0207 0.00145
#3 pt Laplace 0.0209 0.00143
