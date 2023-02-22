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
  // Currently requires data for every node at every time
  // TODO: make this more flexible, probably with spatial interpolation of the
  // PDE solution so we can probe it wherever we like and compare this with the
  // actual data. Alternatively, we could do it the other way round by
  // interpolating the data onto the FD grid. The latter is probably slightly
  // easier? But it's possibly also less correct.

  // Number of nodes in FD discretisation
  int<lower=3> n_node;

  // Distance between nodes in FD discretisation
  real<lower=0> dx;

  // Number of data time points
  int<lower=1> n_time;

  // Starting time
  real<lower=0> t_0;

  // Observed initial conditions for y
  vector[n_node] y_0_obs;

  // Times of data points
  array[n_time] real<lower=0> times;

  // Observed data for y at all other times in each repeat
  array[n_time] vector[n_node] y_obs;

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
  //array[n_time] vector[n_node] y = ode_rk45_tol(rhs, y_0_obs[2:n_node - 1], t_0, times, rel_tol, abs_tol, max_num_steps, n_node, d, a, dx);
  //array[n_time] vector[n_node - 2] y = ode_bdf(rhs, y_0_obs[2:n_node - 1], t_0, times, n_node, d, a, dx, y_bc_left, y_bc_right);
  array[n_time] vector[n_node - 2] y = ode_bdf_tol(rhs, y_0_obs[2:n_node - 1], t_0, times, rel_tol, abs_tol, max_num_steps, n_node, d, a, dx, y_bc_left, y_bc_right);

  //array[n_time] vector[n_node - 2] y =
    //ode_adjoint_tol_ctl(rhs, y_0_obs[2:n_node - 1], t_0, times,
                        //rel_tol/9.0,                         // forward tolerance
                        //rep_vector(abs_tol/9.0, n_node - 2), // forward tolerance
                        //rel_tol/3.0,                         // backward tolerance
                        //rep_vector(abs_tol/3.0, n_node - 2), // backward tolerance
                        //rel_tol,                             // quadrature tolerance
                        //abs_tol,                             // quadrature tolerance
                        //max_num_steps,
                        //150,                                 // number of steps between checkpoints
                        //1,                                   // interpolation polynomial: 1=Hermite, 2=polynomial
                        //2,                                   // solver for forward phase: 1=Adams, 2=BDF 
                        //2,                                   // solver for backward phase: 1=Adams, 2=BDF 
                        //n_node, d, a, dx, y_bc_left, y_bc_right);

  //print(y);
}

model {
  // Likelihood
  // ----------

  for (i in 1:n_time) {
    y_obs[i, 2:n_node - 1] ~ normal(y[i], sigma); // T[0, ];
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
  array[n_output_time + 1] vector[n_node] y_sim;

  y_sim[1] = y_0_obs;

  y_sim[2:n_output_time + 1, 2:n_node - 1] = ode_bdf_tol(rhs, y_0_obs[2:n_node - 1], t_0, output_times, rel_tol, abs_tol, max_num_steps, n_node, d, a, dx, y_bc_left, y_bc_right);

  y_sim[2:n_output_time + 1, 1] = rep_array(y_bc_left, n_output_time);
  y_sim[2:n_output_time + 1, n_node] = rep_array(y_bc_right, n_output_time);

  array[n_output_time + 1] vector[n_node] y_sim_with_noise = y_sim;

  // Add normal noise to all outputs except the ICs and BCs
  for (t in 2:n_output_time + 1) {
    for (n in 2:n_node - 1) {
      y_sim_with_noise[t, n] += normal_rng(0, sigma);
    }
  }
}
