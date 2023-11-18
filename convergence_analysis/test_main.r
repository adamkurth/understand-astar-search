library(testthat)

# Define a test function for density_plot_same
test_density_plot_same <- function() {
  # Create a test plot
  test_plot <- density_plot_same
  
  # Check if the plot is of class ggplot
  expect_true("ggplot" %in% class(test_plot))
  
  # Check if the x-axis label is correct
  expect_equal(ggplot2::xlab(test_plot), "Convergence Time (s)")
  
  # Check if the y-axis label is correct
  expect_equal(ggplot2::ylab(test_plot), "Density")
  
  # Check if the plot title is correct
  expect_equal(ggplot2::ggtitle(test_plot), "Estimated Density Plot of Convergence Time (data_same)")
  
  # Check if the plot theme is minimal
  expect_equal(ggplot2::theme(test_plot), ggplot2::theme_minimal())
}

# Run the test function
test_density_plot_same()library(testthat)

# Define a test function for density_plot_same
test_density_plot_same <- function() {
  # Create a test plot
  test_plot <- density_plot_same
  
  # Check if the plot is of class ggplot
  expect_true("ggplot" %in% class(test_plot))
  
  # Check if the x-axis label is correct
  expect_equal(ggplot2::xlab(test_plot), "Convergence Time (s)")
  
  # Check if the y-axis label is correct
  expect_equal(ggplot2::ylab(test_plot), "Density")
  
  # Check if the plot title is correct
  expect_equal(ggplot2::ggtitle(test_plot), "Estimated Density Plot of Convergence Time (data_same)")
  
  # Check if the plot theme is minimal
  expect_equal(ggplot2::theme(test_plot), ggplot2::theme_minimal())
}

# Define a test function for hypothesis test on dropping number of connections
test_drop_connections <- function() {
  # Perform the hypothesis test
  alpha <- 0.05
  p_value <- perform_hypothesis_test()
  
  # Check if the p-value is less than alpha
  expect_true(p_value < alpha, "The p-value is less than alpha. We can drop the number of connections.")
}

# Run the test functions
test_density_plot_same()
test_drop_connections()# Define a test function for dropping number of connections
test_drop_connections <- function() {
  # Load the required libraries
  library(testthat)
  library(dplyr)
  
  # Perform the hypothesis test
  alpha <- 0.05
  
  # Fit the full model
  full_model <- lm(Convergence.Time ~ Raw.Cost + Total.Cost + Number.of.Nodes + Number.of.Connections, data = data1)
  
  # Fit the reduced model without Number.of.Connections
  reduced_model <- lm(Convergence.Time ~ Raw.Cost + Total.Cost + Number.of.Nodes, data = data1)
  
  # Perform the ANOVA test
  anova_result <- anova(full_model, reduced_model)
  
  # Extract the p-value
  p_value <- anova_result$Pr[2]
  
  # Check if the p-value is less than alpha
  expect_true(p_value < alpha, "The p-value is less than alpha. We can drop the number of connections.")
}

# Run the test function
test_drop_connections()