rm(list=ls())
library(tidyverse)
library(stats)
library(stats4)
library(gridExtra)
library(ggplot2)
library(ggpubr)
library(ggthemes)
library(ggfortify)
library(MASS)

load_data <- function(...) {
    setwd("/Users/adamkurth/Documents/vscode/python_4_fun/A_star/Mountain_Search_Project/convergence_analysis")
    data <- NULL
    filenames <- c(...)
    
    for (filename in filenames) {
        temp_data <- read.csv(filename)
        temp_data <- temp_data[complete.cases(temp_data), ]
        colnames(temp_data) <- c("Convergence.Time", "Raw.Cost", "Total.Cost", "Traffic.Cost", "Path", "Number.of.Nodes", "Number.of.Connections", "Elevation.Function")
        
        if (is.null(data)) {
            data <- temp_data
        } else {
            data <- rbind(data, temp_data)
        }
    }
    return(data)
}

################################################################################
# Read in data
getwd()
same_data_files <- list.files(path = "/Users/adamkurth/Documents/vscode/python_4_fun/A_star/Mountain_Search_Project/convergence_analysis/same_data", pattern = "\\.csv", full.names = TRUE)
data_same <- load_data(same_data_files)
half_data_files <- list.files(path = "/Users/adamkurth/Documents/vscode/python_4_fun/A_star/Mountain_Search_Project/convergence_analysis/half_data", pattern = "\\.csv", full.names = TRUE)
data_half <- load_data(half_data_files)
double_data_files <- list.files(path = "/Users/adamkurth/Documents/vscode/python_4_fun/A_star/Mountain_Search_Project/convergence_analysis/double_data", pattern = "\\.csv", full.names = TRUE)
data_double <- load_data(double_data_files)

################################################################################

# Visualization of convergence time for data_same with jitter
# Scatter plot for data_same
plot_same <- ggplot(data_same, aes(x = Number.of.Nodes, y = Convergence.Time, color = Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
    xlab("Convergence Time (s)") +
    ylab("Number of Nodes") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes (data_same)") +
    theme_minimal()

# Scatter plot for data_half
plot_half <- ggplot(data_half, aes(x = Number.of.Nodes, y = Convergence.Time, color = Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
    xlab("Convergence Time (s)") +
    ylab("Number of Nodes") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes (data_half)") +
    theme_minimal()

# Scatter plot for data_double
plot_double <- ggplot(data_double, aes(x = Number.of.Nodes, y = Convergence.Time, color = Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
    xlab("Convergence Time (s)") +
    ylab("Number of Nodes") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes (data_double)") +
    theme_minimal()


# Histogram for data_same
histogram_same <- ggplot(data_same, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time (data_same)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# Histogram for data_half
histogram_half <- ggplot(data_half, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time (data_half)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# Histogram for data_double
histogram_double <- ggplot(data_double, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time (data_double)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

ggarrange(plot_same, plot_half, plot_double, ncol = 1, nrow = 3)

# Create a grid of plots for all three data groups
grid <- grid.arrange(
    ggarrange(plot_same, histogram_same, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom"),
    ggarrange(plot_half, histogram_half, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom"),
    ggarrange(plot_double, histogram_double, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom"),
    ncol = 1, nrow = 3
)
grid

################################################################################

# Visualization of convergence time for data_same with jitter
# Scatter plot for data_same
plot_same <- ggplot(data_same, aes(x = Number.of.Nodes, y = Convergence.Time, color = Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
    xlab("Number of Nodes") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes (data_same)") +
    theme_minimal()

# Scatter plot for data_half
plot_half <- ggplot(data_half, aes(x = Number.of.Nodes, y = Convergence.Time, color = Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
    xlab("Number of Nodes") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes (data_half)") +
    theme_minimal()

# Scatter plot for data_double
plot_double <- ggplot(data_double, aes(x = Number.of.Nodes, y = Convergence.Time, color = Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.2, height = 0.2)) +
    xlab("Number of Nodes") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Convergence Time vs Number of Nodes (data_double)") +
    theme_minimal()


# Histogram for data_same
histogram_same <- ggplot(data_same, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time (data_same)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# Histogram for data_half
histogram_half <- ggplot(data_half, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time (data_half)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# Histogram for data_double
histogram_double <- ggplot(data_double, aes(x = Convergence.Time)) +
    geom_histogram(bins = 15, fill = "#0072B2", color = "black") +
    xlab("Convergence Time (s)") +
    ylab("Frequency") +
    ggtitle("Histogram of Convergence Time (data_double)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())


grid <- grid.arrange(plot_same, plot_half, plot_double, ncol = 3, nrow = 1)
grid

# Create a grid of plots for all three data groups
grid <- grid.arrange(
    ggarrange(plot_same, histogram_same, ncol = 3, nrow = 1, common.legend = TRUE, legend = "bottom"),
    ggarrange(plot_half, histogram_half, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom"),
    ggarrange(plot_double, histogram_double, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom"),
    ncol = 1, nrow = 3
)
grid

################################################################################
# Density plot for data_half
density_plot_half <- ggplot(data_half, aes(x = Convergence.Time)) +
    geom_density(fill = "#0072B2", alpha = 0.5) +
    xlab("Convergence Time (s)") +
    ylab("Density") +
    ggtitle("Estimated Density Plot of Convergence Time (data_half)") +
    theme_minimal()

# Density plot for data_double
density_plot_double <- ggplot(data_double, aes(x = Convergence.Time)) +
    geom_density(fill = "#0072B2", alpha = 0.5) +
    xlab("Convergence Time (s)") +
    ylab("Density") +
    ggtitle("Estimated Density Plot of Convergence Time (data_double)") +
    theme_minimal()

# Density plot for data_same
density_plot_same <- ggplot(data_same, aes(x = Convergence.Time)) +
    geom_density(fill = "#0072B2", alpha = 0.5) +
    xlab("Convergence Time (s)") +
    ylab("Density") +
    ggtitle("Estimated Density Plot of Convergence Time (data_same)") +
    theme_minimal()

# Create a grid of plots for all three data groups
grid <- grid.arrange(
    density_plot_half,
    density_plot_double,
    density_plot_same,
    ncol = 1, nrow = 3
)
grid

################################################################################

# Exploring whether traffic cost is a good predictor of convergence time: ANS: no

# Scatter plot of Total Traffic Cost vs Convergence Time for data_same
scatter_plot_same <- ggplot(data_same, aes(x = Total.Cost, y = Convergence.Time)) +
    geom_point(color = "#0072B2", size = 3) +
    xlab("Total Traffic Cost") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Total Traffic Cost vs Convergence Time (data_same)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# Scatter plot of Total Traffic Cost vs Convergence Time for data_half
scatter_plot_half <- ggplot(data_half, aes(x = Total.Cost, y = Convergence.Time)) +
    geom_point(color = "#0072B2", size = 3) +
    xlab("Total Traffic Cost") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Total Traffic Cost vs Convergence Time (data_half)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# Scatter plot of Total Traffic Cost vs Convergence Time for data_double
scatter_plot_double <- ggplot(data_double, aes(x = Total.Cost, y = Convergence.Time)) +
    geom_point(color = "#0072B2", size = 3) +
    xlab("Total Traffic Cost") +
    ylab("Convergence Time (s)") +
    ggtitle("Scatter Plot of Total Traffic Cost vs Convergence Time (data_double)") +
    theme_minimal() +
    theme(plot.title = element_text(color = "#0072B2", size = 14, face = "bold"),
          axis.title.x = element_text(color = "#0072B2", size = 12),
          axis.title.y = element_text(color = "#0072B2", size = 12),
          axis.text = element_text(color = "#0072B2", size = 10),
          panel.background = element_rect(fill = "#F0F0F0"),
          panel.grid.major = element_line(color = "#D3D3D3"),
          panel.grid.minor = element_blank())

# scatter plots for all three data frames
grid <- grid.arrange(
    scatter_plot_same,
    scatter_plot_half,
    scatter_plot_double,
    ncol = 1, nrow = 3
)
grid

################################################################################

# Create the histograms for data_same
histogram_total_same <- ggplot(data_same, aes(x = Total.Cost)) +
    geom_histogram(bins = 30, fill = "#0072B2", color = "black") +
    xlab("Total Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Total Cost (data_same)") +
    theme_minimal()

histogram_raw_same <- ggplot(data_same, aes(x = Raw.Cost)) +
    geom_histogram(bins = 30, fill = "#E69F00", color = "black") +
    xlab("Raw Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Raw Cost (data_same)") +
    theme_minimal()

histogram_traffic_same <- ggplot(data_same, aes(x = Traffic.Cost)) +
    geom_histogram(bins = 30, fill = "#009E73", color = "black") +
    xlab("Traffic Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Traffic Cost (data_same)") +
    theme_minimal()

# Create the histograms for data_half

histogram_total_half <- ggplot(data_half, aes(x = Total.Cost)) +
    geom_histogram(bins = 30, fill = "#0072B2", color = "black") +
    xlab("Total Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Total Cost (data_half)") +
    theme_minimal()

histogram_raw_half <- ggplot(data_half, aes(x = Raw.Cost)) +
    geom_histogram(bins = 30, fill = "#E69F00", color = "black") +
    xlab("Raw Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Raw Cost (data_half)") +
    theme_minimal()

histogram_traffic_half <- ggplot(data_half, aes(x = Traffic.Cost)) +
    geom_histogram(bins = 30, fill = "#009E73", color = "black") +
    xlab("Traffic Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Traffic Cost (data_half)") +
    theme_minimal()

# Create the histograms for data_double
histogram_total_double <- ggplot(data_double, aes(x = Total.Cost)) +
    geom_histogram(bins = 30, fill = "#0072B2", color = "black") +
    xlab("Total Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Total Cost (data_double)") +
    theme_minimal()

histogram_raw_double <- ggplot(data_double, aes(x = Raw.Cost)) +
    geom_histogram(bins = 30, fill = "#E69F00", color = "black") +
    xlab("Raw Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Raw Cost (data_double)") +
    theme_minimal()

histogram_traffic_double <- ggplot(data_double, aes(x = Traffic.Cost)) +
    geom_histogram(bins = 30, fill = "#009E73", color = "black") +
    xlab("Traffic Cost") +
    ylab("Frequency") +
    ggtitle("Histogram of Traffic Cost (data_double)") +
    theme_minimal()

grid <- grid.arrange(
    histogram_total_same, histogram_raw_same, histogram_traffic_same,
    histogram_total_half, histogram_raw_half, histogram_traffic_half,
    histogram_total_double, histogram_raw_double, histogram_traffic_double,
    ncol = 3, nrow = 3
)

# Display the grid
grid

################################################################################

data1 <- rbind(data_same, data_half, data_double) 
numeric_data <- data1[sapply(data1, is.numeric)]
numeric_data <- subset(numeric_data, select = -Elevation.Function)

# Fit the full model
full_model <- lm(Convergence.Time ~ ., data = numeric_data)

# Perform stepwise regression using BIC for model selection
# 'k = log(nrow(numeric_data))' sets the penalty per parameter as log(n), which is used for BIC
stepwise_model <- stepAIC(full_model, direction = "both", k = log(nrow(numeric_data)))


m1 = lm(Convergence.Time ~ Total.Cost + Traffic.Cost + Number.of.Nodes, data=data1)
anova(m1)

data1$Number.of.Nodes.2 = (data1$Number.of.Nodes)^2
m2 = lm(Convergence.Time ~ Total.Cost + Traffic.Cost + Number.of.Nodes + Number.of.Nodes.2, data=data1)

anova(m2)
summary(m2)
anova(m1,m2)

numeric_data <- data1[sapply(data1, is.numeric)]
numeric_data <- subset(numeric_data, select = -Elevation.Function)
full.model = lm(Convergence.Time ~ ., data = numeric_data)
stepwise_model <- stepAIC(full.model, direction = "both", k = log(nrow(numeric_data)))


# the reduced model is a better fit, so we will use that
poly_model <- m2

# we see that Raw.Cost and Number.of.Nodes are the two lowest p-values, so we will use those two

par(mfrow=c(2,2))
plot(poly_model)
par(mfrow=c(1,1))


## residual plot for poly_model
residuals_df <- data.frame(
  Fitted = fitted(poly_model),
  Residuals = residuals(poly_model)
)


ggplot(residuals_df, aes(x = Fitted, y = Residuals)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") + 
  geom_jitter(aes(color = Residuals), width = 0.1, height = 0) +  
  theme_minimal() + 
  labs(x = "Fitted Values", y = "Residuals", title = "Residual Plot with Jitter") +
  scale_color_gradientn(colours = rainbow(4)) +  
  theme(
    plot.title = element_text(hjust = 0.5),  
    axis.title.x = element_text(vjust = -0.2), 
    axis.title.y = element_text(vjust = 2)  
)
ggsave("residual_plot.png", width = 8, height = 4, dpi = 300)  



# Create a data frame with the fitted values and residuals
residuals_df <- data.frame(
    Fitted = fitted(poly_model),
    Residuals = residuals(poly_model)
)

# Create a scatter plot of fitted values versus actual values with jitter
a <- ggplot(residuals_df, aes(x = Fitted, y = data1$Convergence.Time)) +
    geom_point(position = position_jitter(width = 0.02, height = 0.02) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    labs(x = "Fitted Values", y = "Actual Values", title = "Scatter Plot of Fitted Values vs Actual Values") +
    theme_minimal()
b <- ggplot(residuals_df, aes(x = Fitted, y = data1$Convergence.Time)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    labs(x = "Fitted Values", y = "Actual Values", title = "Scatter Plot of Fitted Values vs Actual Values") +
    theme_minimal()

ggarrange(a, b, ncol = 1, nrow = 2)

car::influencePlot(poly_model)
anova(poly_model)
summary(poly_model)
summary(poly_model)$r.squared
summary(poly_model)$adj.r.squared
summary(poly_model)$coefficients
detach(data1)