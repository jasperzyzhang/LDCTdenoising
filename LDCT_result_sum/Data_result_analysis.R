#Formalize the result and Data Visualization
#load pacakges needed
library(Rmisc)
library(plotly)
library(tidyverse)
library(reshape2)
library(dplyr)
library(gmodels)
library(ggplot2)
library(readr)

#define result path
resultpath <- "~/Documents/Projects/LDCT_code/result_visualization/LDCT_result_sum/"

wiener <- read_csv(paste(resultpath,"wiener.csv",sep = ""), )
admm_tv <- read_csv(paste(resultpath,"admm_tv.csv",sep = ""), )
admm_dncnn <- read_csv(paste(resultpath,"admm_dncnn.csv",sep = ""), )
ae <- read_csv(paste(resultpath,"ae.csv",sep = ""), )
dncnn <- read_csv(paste(resultpath,"dncnn.csv",sep = ""), )

all = rbind(wiener,admm_tv,admm_dncnn,ae,dncnn)
#filtering the results needed
all = all[all$type =="denoised",]
all = all[all$noise !="poiimg",]

#summary of PSNR and MSE
psnr_ci = group.CI(PSNR ~  model + noise + type ,data = all, ci = 0.95)
mse_ci = group.CI(MSE ~  model + noise + type ,data = all, ci = 0.95)

#merged result matrix
mm = merge(psnr_ci,mse_ci)

mean_sum = mm[,c(1,2,5,8)]
names(mean_sum) = c("model","noise","PSNR","MSE")
mean_sum$PSNR = round(mean_sum$PSNR,digits = 2)
mean_sum$MSE = round(mean_sum$MSE,digits = 5)

#redefine
noise_name = c("Gaussian: N(0,10)", "Gaussian: N(0,50)", "Poisson: Poi(50)")

psnr = mean_sum[,c(1,2,3)]
psnr_final = spread(psnr, model, PSNR)
col_order = c(1,6,3,2,4,5)
psnr_final = psnr_final[,col_order]
psnr_final$noise = noise_name
#save psnr result
write.csv(psnr_final, "psnr.csv", row.names=FALSE)

mse = mean_sum[,c(1,2,4)]
mse_final = spread(mse, model, MSE)
mse_final = mse_final[,col_order]
mse_final$noise = noise_name
#save mse result
write.csv(mse_final, "mse.csv", row.names=FALSE)

knitr::kable(mse_final)



