rm(list=ls())
library(tidyverse)
library(stats)
library(stats4)
library(gridExtra)
library(ggplot2)
library(ggpubr)
library(ggthemes)
library(ggfortify)



## using ggplot for experimentation
residuals_vs_fitted <- function(model) {
    ggplot(model, aes_string(x = ".fitted", y = ".resid")) +
        geom_point() +
        geom_hline(yintercept = 0, linetype = "dashed") +
        labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs Fitted Values")
}

normal_qq <- function(model) {
    ggplot(model, aes(sample = .stdresid)) +
        stat_qq() +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
        labs(title = "Normal Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles")
}

scale_location <- function(model) {
    ggplot(model, aes(.fitted, sqrt(abs(.stdresid)))) +
        geom_point() +
        geom_hline(yintercept = 0, linetype = "dashed") +
        labs(x = "Fitted Values", y = "Square Root of Standardized Residuals", title = "Scale-Location Plot")
}

leverage_plot <- function(model) {
    ggplot(model, aes(.hat, .stdresid)) +
        geom_point() +
        labs(x = "Leverage", y = "Standardized Residuals", title = "Residuals vs Leverage")
}


# Read in data
getwd()
setwd("/Users/adamkurth/Documents/vscode/python_4_fun/A_star/Mountain_Search_Project/convergence_analysis")
data1 <- read.csv("all_results_df_same.csv")
data1 <- data1[complete.cases(data1), ]
colnames(data1) = c("Convergence.Time", "Raw.Cost", "Total.Cost", "Traffic.Cost", "Path", "Number.of.Nodes", "Number.of.Connections", "Elevation.Function", "Num_Nodes", "Num_Connections")
# Remove columns "Num_Nodes" and "Num_Connections"
data1 <- data1[, -c(9, 10)]
data1 <- as.data.frame(data1)
attach(data1)   

# Visualization of convergence time
plot <- ggplot(data1, aes(x = Convergence.Time, y = Number.of.Nodes)) +
    geom_point() +
    xlab("Convergence Time (s)") +
    ylab("Number of Nodes") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes") +
    theme_minimal()

histogram <- ggplot(data1, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time") +
    theme_minimal()

density_plot <- ggplot(data1, aes(x = Convergence.Time)) +
    geom_density(fill = "#0072B2", alpha = 0.5) +
    xlab("Convergence Time (s)") +
    ylab("Density") +
    ggtitle("Estimated Density Plot of Convergence Time") +
    theme_minimal()

ggarrange(plot, histogram, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")
ggarrange(plot, density_plot, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")


# more natural visualization of convergence time
plot <- ggplot(data1, aes(x = Number.of.Nodes, y = Convergence.Time)) +
    geom_point() +
    xlab("Number of Nodes") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes") +
    theme_minimal()
plot

# visualize the skew of total and raw cost
a <- ggplot(data1, aes(x = Total.Cost)) +
    geom_histogram(bins = 30, fill = "#0072B2", color = "black") +
    xlab("Total Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Total Cost") +
    theme_minimal()

b <- ggplot(data1, aes(x = Raw.Cost)) +
    geom_histogram(bins = 30, fill = "#E69F00", color = "black") +
    xlab("Raw Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Raw Cost") +
    theme_minimal()

c <- ggplot(data1, aes(x = Traffic.Cost)) +
    geom_histogram(bins = 30, fill = "#009E73", color = "black") +
    xlab("Traffic Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Traffic Cost") +
    theme_minimal()

ggarrange(a, b, c, ncol = 3, nrow = 1, common.legend = TRUE, legend = "bottom")

poly_model <- lm(Convergence.Time ~ poly(Number.of.Nodes,2), data=data1)

a = residuals_vs_fitted(poly_model)
b = normal_qq(poly_model)
c = scale_location(poly_model)
d = leverage_plot(poly_model)
ggarrange(a,b,c,d, ncol = 2, nrow = 2, common.legend = TRUE, legend = "bottom")

car::influencePlot(poly_model)
anova(poly_model)
summary(poly_model)
summary(poly_model)$r.squared
summary(poly_model)$adj.r.squared




detach(data1)


## for original data
data = read.csv('../data.csv')
plot(y=data$Convergence.Time, x=data$Number.of.Nodes.in.Path, df=data, x_label="Number of Nodes", y_label="Convergence Time (s)")
plot(y=data$Convergence.Time, x=data$Cost, df=data, x_label="Cost", y_label="Convergence Time (s)")
plot(y=data$Convergence.Time, x=data$Number.of.Connections, df=data, x_label="Number of Connections", y_label="Convergence Time (s)")
