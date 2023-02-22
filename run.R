library(data.table)
library(cmdstanr)
library(ggplot2)
library(magrittr)
library(bayesplot)
library(posterior)
library(cowplot)
library(ggfan)

options(mc.cores = 16)

# Read the raw data into a single data.table
data_raw <- fread("synthetic_data.csv")
times_raw <- fread("synthetic_data_times.csv")

# Number of FD nodes in FD discretisation
n_node <- 101

# Distance between FD nodes - assume domain length of 1
dx <- 1.0 / (n_node - 1)

data_raw_t <- transpose(data_raw)
data_raw_t[, x := seq(0, 1, length.out = n_node)]
data_raw_long <- data_raw_t %>% melt(id.vars = "x")
data_raw_long[, time := rep(times_raw[, V1], each = n_node)]
data_raw_long[, variable := NULL]
setnames(data_raw_long, "value", "y")
setcolorder(data_raw_long, c("time", "x", "y"))

# Plot the raw data
ggplot(data_raw_long, aes(x = x, y = y, colour = time)) +
  geom_line(linewidth = 0.5, linetype = "dotted") +
  geom_point(size = 1) +
  facet_wrap(vars(factor(time)))

# Get the number of observations
n_time <- nrow(times_raw)

# Extract the times and data, excluding the initial time/conditions
times <- times_raw[, .SD[-1]][, V1]
y_obs <- data_raw[, .SD[-1]]

# Extract the ICs
y_0_obs <- data_raw[, .SD[1]] %>% transpose %>% .[, V1]

# Subtract one from the data set length
n_time <- n_time - 1

# Initial time
t_0 <- 0

# Output times for model posterior draws
n_output_time <- 50
max_time <- 1.0

# Stan doesn't allow outputting the ode solution at time == initial time, so do
# one more time than required starting at t = 0, then remove the first one so
# we end up with `n_output_data` times, but without t = 0
output_times_full <- seq(0, 1, length.out = n_output_time + 1) * max_time
output_times <- output_times_full[-1]

# Construct the data list to send to Stan
data_list <- list(
  n_node = n_node,
  dx = dx,
  n_time = n_time,
  t_0 = t_0,
  y_0_obs = y_0_obs,
  times = times,
  y_obs = y_obs,
  rel_tol = 1e-3,
  abs_tol = 1e-6,
  max_num_steps = 1000,
  n_output_time = n_output_time,
  output_times = output_times
)

init <- 1.0

# Compile the model
mod <- cmdstan_model("model.stan")

# Estimate parameters
fit <- mod$sample(
  data = data_list,
  iter_sampling = 1000,
  init = init,
  chains = 4
)

#fit$save_object("fit_n_node=101_n_time=2.RDS")

#fit <- readRDS("fit_n_node=101_n_time=2.RDS")

mcmc_recover_intervals(fit$draws(c("d", "a", "sigma")), c(0.05, 0.5, 0.01))
mcmc_dens(fit$draws(c("d", "a", "sigma")))
mcmc_dens_overlay(fit$draws(c("d", "a", "sigma")))
mcmc_pairs(fit$draws(c("d", "a", "sigma")))

# Posterior predictive checks
d_r <- fit$draws() %>% as_draws_rvars()

# Get the posterior draws
y_posterior_draws <- d_r$y_sim_with_noise %>%
  draws_of %>%
  as.data.table

# Proper column names, times and coords
setnames(y_posterior_draws, c("iter", "timestep", "x", "y"))
y_posterior_draws <- y_posterior_draws %>%
  .[, iter := as.integer(iter)] %>%
  .[order(iter, timestep)] %>%
  .[, time := output_times_full[timestep]] %>%
  .[, x := (x - 1) / (n_node - 1)]

# Calculate quantiles
y_posterior_quantiles <- y_posterior_draws[
  ,
  .(
    q10 = quantile(y, 0.1),
    median = median(y),
    q90 = quantile(y, 0.9)
  ),
  by = .(x, time)
]

# Plot posterior draw quantiles with data overlaid

# Spatial profiles for each output time
ggplot(
  y_posterior_quantiles,
  aes(x = x, y = median, colour = time, fill = time)
) +
  geom_ribbon(aes(ymin = q10, ymax = q90, colour = NULL), alpha = 0.2) +
  geom_line() +
  geom_line(data = data_raw_long, aes(y = y), linewidth = 0.5, linetype = "dotted") +
  geom_point(data = data_raw_long, aes(y = y), shape = 21, colour = "black", size = 1) +
  facet_wrap(vars(factor(time)))

# Value at each node as a time series
ggplot(
  y_posterior_quantiles,
  aes(x = time, y = median, colour = x, fill = x)
) +
  geom_ribbon(aes(ymin = q10, ymax = q90, colour = NULL), alpha = 0.2) +
  geom_line() +
  geom_line(data = data_raw_long, aes(y = y), linewidth = 0.5, linetype = "dotted") +
  geom_point(data = data_raw_long, aes(y = y), shape = 21, colour = "black", size = 1) +
  facet_wrap(vars(factor(x)))
