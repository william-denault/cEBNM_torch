library(flashier)
library(ggplot2)
library(reshape2)

set.seed(1)

n_reps <- 10
rmse_results <- data.frame(
  rep = integer(),
  method = character(),
  rmse = numeric()
)

for (i in 1:n_reps) {
  set.seed(i)
  
  N <- 1000
  P <- 200
  
  # -------------------------
  # True loadings L (N x 7), block/binary indicator structure
  # -------------------------
  L_true <- matrix(0, nrow = N, ncol = 7)
  L_true[, 1] <- 1
  L_true[1:500, 2] <- 1
  L_true[501:1000, 3] <- 1
  L_true[1:250, 4] <- 1
  L_true[251:500, 5] <- 1
  L_true[501:750, 6] <- 1
  L_true[751:1000, 7] <- 1
  
  # -------------------------
  # True factors f_mat (7 x P): sparse via binary masks * N(0,1)
  # -------------------------
  t0  <- rbinom(P, 1, 0.5)
  t1  <- rbinom(P, 1, 0.5)
  t2  <- rbinom(P, 1, 0.5)
  t11 <- rbinom(P, 1, 0.5)
  
  a0     <- t0  * rnorm(P)
  b1     <- t1  * rnorm(P)
  b2     <- t2  * rnorm(P)
  delta11 <- t11 * rnorm(P)
  delta12 <- t11 * rnorm(P)
  delta21 <- t11 * rnorm(P)
  delta22 <- t11 * rnorm(P)
  
  f_mat <- rbind(a0, b1, b2, delta11, delta12, delta21, delta22)  # 7 x P
  
  # -------------------------
  # Signal and noisy data
  # -------------------------
  X_true <- L_true %*% f_mat
  noise  <- matrix(rnorm(N * P), nrow = N, ncol = P) * 0.5 * 2.5
  Z <- X_true + noise
  
  # -------------------------
  # Fit flash: GB prior on L, point_normal on F (analogous to "cgb")
  # -------------------------
  fit_gb <- flash(
    Z,
    greedy_Kmax = 10,
    ebnm_fn = c(ebnm_generalized_binary, ebnm_point_normal),
    backfit = TRUE,
    verbose = 0
  )
  
  X_hat_gb <- fit_gb$L_pm %*% t(fit_gb$F_pm)
  rmse_gb <- sqrt(mean((X_true - X_hat_gb)^2))
  
  # -------------------------
  # Fit flash: GB prior on L, ash on F (analogous to "cgb_sharp_2")
  # -------------------------
  fit_gb_ash <- flash(
    Z,
    greedy_Kmax = 10,
    ebnm_fn = c(ebnm_generalized_binary, ebnm_ash),
    backfit = TRUE,
    verbose = 0
  )
  
  X_hat_gb_ash <- fit_gb_ash$L_pm %*% t(fit_gb_ash$F_pm)
  rmse_gb_ash <- sqrt(mean((X_true - X_hat_gb_ash)^2))
  
  rmse_results <- rbind(
    rmse_results,
    data.frame(rep = i, method = "GB + point_normal", rmse = rmse_gb),
    data.frame(rep = i, method = "GB + ash",          rmse = rmse_gb_ash)
  )
  
  cat("rep", i, "done\n")
}

# -------------------------
# Summary
# -------------------------
aggregate(rmse ~ method, data = rmse_results, FUN = function(x) c(mean = mean(x), sd = sd(x)))

ggplot(rmse_results, aes(x = method, y = rmse)) +
  geom_boxplot() +
  geom_jitter(width = 0.1, alpha = 0.6) +
  theme_minimal() +
  labs(title = "Reconstruction RMSE across replicates", y = "RMSE", x = "")
