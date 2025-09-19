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
n_iter <- 200 

rmse <- numeric(n_iter)

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
  
  Zhat <- fitted(fit)                       # n x p
  rmse[i] <- sqrt(mean((Zhat - S)^2))
}


quantile(rmse)
# --- Summary ---
cat(sprintf("RMSE over %d sims: mean = %.4f, sd = %.4f\n",
            n_iter, mean(rmse), sd(rmse)))

# Optional: quick plot
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  df <- data.frame(rmse = rmse)
  print(
    ggplot(df, aes(rmse)) +
      geom_histogram(bins = 30) +
      theme_minimal() +
      labs(title = "flashier RMSE vs. true rank-1 signal",
           x = "RMSE", y = "Count")
  )
}
