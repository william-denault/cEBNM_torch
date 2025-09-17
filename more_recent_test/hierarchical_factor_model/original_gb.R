library(reshape2)

library(ggplot2)
library(flashier)

library(RcppCNPy)

path <- "C:/Document/Serieux/Travail/python_work/cEBNM_torch/more_recent_test/hierarchical_factor_model"

Z <- as.matrix(read.table(file.path(path, "Z_matrix.txt"), header=FALSE))
dim(Z)

sum(is.na(Z))
### Laplace -----
fit_greedy_laplace <- flash(
  Z, 
  greedy_Kmax = 10,  
  ebnm_fn = c(ebnm_generalized_binary, ebnm_point_laplace), 
  backfit = TRUE, 
  verbose = 0
)
L_df_laplace <- melt(fit_greedy_laplace$L_pm)
P_laplace= ggplot(L_df_laplace, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  ggtitle("GB+ Laplace")+
  labs(x="Component", y="Sample", fill="Value")

P_laplace





fit_greedy_norm <- flash(
  Z, 
  greedy_Kmax = 10,  
  ebnm_fn = c(ebnm_generalized_binary, ebnm_point_normal), 
  backfit = TRUE, 
  verbose = 0
)
L_df_norm <- melt(fit_greedy_norm$L_pm)
P_point_norm= ggplot(L_df_norm, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  ggtitle("GB+ point norm")+
  labs(x="Component", y="Sample", fill="Value")





fit_greedy_ash= fit_greedy_ash <- flash(
  Z, 
  greedy_Kmax = 10,  
  ebnm_fn = c(ebnm_generalized_binary, ebnm_ash), 
  backfit = TRUE, 
  verbose = 0
)
L_df_ash <- melt(fit_greedy_ash$L_pm)
P_ash=ggplot(L_df_ash, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  ggtitle("GB+ ash")+
  labs(x="Component", y="Sample", fill="Value")






#fit_greedy_horseshoe <- flash(
#  Z, 
#  greedy_Kmax = 10,  
#  ebnm_fn = c(ebnm_generalized_binary, ebnm_horseshoe), 
#  backfit = TRUE, 
#  verbose = 0
#)
#L_df_horseshoe <- melt(fit_greedy_horseshoe$L_pm)
#P_horseshoe= ggplot(L_df_horseshoe, aes(x=Var2, y=Var1, fill=value)) +
#  geom_tile() +
#  scale_fill_viridis_c() +
#  theme_minimal() +
  
#  ggtitle("GB+ horseshoe")+
#  labs(x="Component", y="Sample", fill="Value")





fit_greedy_unimodal <- flash(
  Z, 
  greedy_Kmax = 10,  
  ebnm_fn = c(ebnm_generalized_binary, ebnm_unimodal), 
  backfit = TRUE, 
  verbose = 0
)
L_df_unimodal <- melt(fit_greedy_unimodal$L_pm)
P_unimodal= ggplot(L_df_unimodal, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  
  ggtitle("GB+ unimoda")+
  labs(x="Component", y="Sample", fill="Value")



library(gridExtra)

grid.arrange(P_laplace,
             P_point_norm,
             P_ash,
             P_unimodal, ncol=4  )



path <- "C:/Document/Serieux/Travail/python_work/cEBNM_torch/more_recent_test/hierarchical_factor_model"

X <- as.matrix(read.table(file.path(path, "X_matrix.txt"), header=FALSE))
dim(X)

mean( (c(X)- c(fit_greedy_laplace$L_pm%*%t(fit_greedy_laplace$F_pm)))^2)
mean( (c(X)- c(fit_greedy_ash$L_pm%*%t(fit_greedy_ash$F_pm)))^2)

mean( (c(X)- c(fit_greedy_norm$L_pm%*%t(fit_greedy_norm$F_pm)))^2)
mean( (c(X)- c(fit_greedy_unimodal$L_pm%*%t(fit_greedy_unimodal$F_pm)))^2)
