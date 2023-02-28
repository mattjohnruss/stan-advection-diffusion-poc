functions {
  vector rhs(real t,
             vector y,
             int n_node,
             real d,
             real a,
             real dx,
             real y_bc_left,
             real y_bc_right) {
    vector[n_node - 2] dydt;

    vector[n_node] y_all;
    y_all[1] = y_bc_left;
    y_all[2:n_node - 1] = y;
    y_all[n_node] = y_bc_right;

    // Bulk
    // Note the loop range and shifted index on the LHS. We could loop over
    // 1:n_node - 2 but then we'd have to shift the indices on the RHS which
    // would be a lot more confusing
    for (i in 2:n_node - 1) {
      dydt[i - 1] = d * (y_all[i - 1] - 2.0 * y_all[i] + y_all[i + 1]) / dx^2
        - a * (y_all[i + 1] - y_all[i]) / dx;
    }

    return dydt;
  }

  real lerp(real v_1, real v_2, real s) {
    return (1.0 - s) * v_1 + s * v_2;
  }
}

data {
  // Number of nodes in FD discretisation
  int<lower=3> n_node_sim;

  // Number of spatial data points per time sample
  int<lower=1> n_node_data;

  // Distance between nodes in FD discretisation
  real<lower=0> dx;

  // Number of data time points
  int<lower=1> n_time;

  // Starting time
  real<lower=0> t_0;

  // Initial conditions for y
  // Either needs to be interpolated onto the computational grid before being
  // provided to Stan as `data`, or set the ICs directly in `transformed data`.
  //vector[n_node_sim] y_0;

  // Times of data points
  array[n_time] real<lower=0> times;

  // Observed data for y at all other times in each repeat
  // Note: this includes the boundaries
  array[n_time] vector[n_node_data] y_obs;

  // Lookup tables for interpolating solution onto data grid
  array[n_node_data, 2] int node_indices;
  array[n_node_data] real s;

  // ODE solver parameters
  real rel_tol;
  real abs_tol;
  int max_num_steps;

  // Number of output data time points
  int<lower=1> n_output_time;

  // Times of output data points
  array[n_output_time] real<lower=0> output_times;
}

transformed data {
  // Initial conditions at the computational nodes - see above
  vector[n_node_sim] y_0;
  for (i in 1:n_node_sim) {
    real x = (i - 1.0) / (n_node_sim - 1.0);
    y_0[i] = x * (1.0 - x);
  }

  real y_bc_left = 0.0;
  real y_bc_right = 0.0;
}

parameters {
  // Diffusivity
  real<lower=0> d;

  // Advection velocity
  real<lower=0> a;

  // Noise
  real<lower=0> sigma;
}

transformed parameters {
  // Solution, including boundaries (required for interp)
  array[n_time] vector[n_node_sim] y;

  // Set the boundary values
  y[1:n_time, 1] = rep_array(0.0, n_time);
  y[1:n_time, n_node_sim] = rep_array(0.0, n_time);

  // Solve the ODE system for the bulk values and set them
  y[1:n_time, 2:n_node_sim - 1] = ode_bdf_tol(rhs, y_0[2:n_node_sim - 1], t_0, times, rel_tol, abs_tol, max_num_steps, n_node_sim, d, a, dx, y_bc_left, y_bc_right);

  array[n_time] vector[n_node_data - 2] y_interp;

  // Interpolate the solution onto the data grid
  for (i in 1:n_time) {
    for (j in 2:n_node_data - 1) {
      y_interp[i, j - 1] =
        lerp(y[i, node_indices[j, 1] + 1], y[i, node_indices[j, 2] + 1], s[j]);
    }
  }
}

model {
  // Likelihood
  // ----------

  for (i in 1:n_time) {
    y_obs[i, 2:n_node_data - 1] ~ normal(y_interp[i], sigma);
  }

  // Priors
  // ------

  // Diffusivity
  d ~ normal(0.01, 1.0) T[0, ];

  // Advection velocity
  a ~ normal(0.01, 1.0) T[0, ];

  // Noise
  sigma ~ normal(0.1, 0.1) T[0, ];
}

generated quantities {
  array[n_output_time + 1] vector[n_node_sim] y_sim;

  y_sim[1] = y_0;

  y_sim[2:n_output_time + 1, 2:n_node_sim - 1] = ode_bdf_tol(rhs, y_0[2:n_node_sim - 1], t_0, output_times, rel_tol, abs_tol, max_num_steps, n_node_sim, d, a, dx, y_bc_left, y_bc_right);

  y_sim[2:n_output_time + 1, 1] = rep_array(y_bc_left, n_output_time);
  y_sim[2:n_output_time + 1, n_node_sim] = rep_array(y_bc_right, n_output_time);

  array[n_output_time + 1] vector[n_node_sim] y_sim_with_noise = y_sim;

  // Add normal noise to all outputs except the ICs and BCs
  for (t in 2:n_output_time + 1) {
    for (n in 2:n_node_sim - 1) {
      y_sim_with_noise[t, n] += normal_rng(0, sigma);
    }
  }
}
