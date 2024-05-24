/* As a second step, we use the lattice Boltzmann (LB) technique to simulate the problem of 2D Poiseuille flow in rectangular coordinates. (This is the problem of viscous flow between two rectangular
 * plates due to a constant pressure drop in the x-direction (dp/dx = C for constant C), where both the top and bottom walls are fixed. We use no-slip boundary conditions, and specify the value of
 * dp/dx.) For a channel of length L, a viscosity of μ, and a pressure drop of dp/dx; the velocity is given by u(y) = (dp/dx) * (y² - L²)/2μ.
 *
 * As with the previous script (couette_wetnode_noneq_BCs_functions.cpp), we heavily reference Krüger et al., The Lattice Boltzmann Method, Springer Nature, Cham, Switzerland, 2017. We also reference
 * that previous script heavily, including in the comments.
 *
 * For the LB simulation, we use the following setup:
 *   Collision operator:    BGK approximation
 *   Lattice structure:     D2Q9
 *   1st order BCs:         Periodic in x with a pressure drop; wet-node bounce-back (c.f. Krüger Pg. 177) in y at walls
 *   2nd order (noneq) BCs: Zou-He (nonequilibrium bounce back) (c.f. Krüger Pgs. 196-199). The equations for the nonequilibrium extrapolation method (c.f. Krüger Pgs. 194-195) are also included for
 *                          reference.
 *   Relaxation scheme:     Single relaxation time
 *   Added forces:          None
 *   Number of phases:      1
 *
 * Note that the same outstanding changes that need to be made in couette_wetnode_noneq_BCs_functions.cpp also need to be made here. */


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <time.h>


/* Here, we define the 1D, 2D, and 3D floating point array classes as specific types. The rationale for this is explained further in couette_wetnode_noneq_BCs_functions.cpp. */
typedef std::vector<double> vect1D_float;
typedef std::vector<std::vector<double>> array2D_float;
typedef std::vector<std::vector<std::vector<double>>> array3D_float;


/* Here, we define the simulation parameters. We note that the system is periodic in x (with a pressure drop along x). We also note that the first and last nodes in the x-direction (x = 0 and
 * x = x_len - 1) are virtual nodes; however, all of the notes in the y-direction are real nodes. */
const int x_len = 42;             // number of x grid points (including the virtual nodes at x = 0 and x = x_len - 1).
const int y_len = 21;             // number of y grid points
const int t_steps = 10000001;     // number of time steps. Since I update the densities and velocities at the beginning of the simulation, I'm doing one extra timestep.
const double tau = 0.9;           // BGK relaxation time
const double omega = 1.0/tau;     // BGK inverse relaxation time (lowercase omega, NOT capital omega)
const double rho_start = 1.0;     // initial (starting) density
const double cs_sq = 1.0/3.0;     // speed of sound squared in lattice units
const double u_wall_top = 0.0;    // top wall velocity
const double u_wall_bottom = 0.0; // bottom wall velocity
const double ux_init_glob = 0.0;  // global initial velocity in the x-direction
const double uy_init_glob = 0.0;  // global initial velocity in the y-direction
const double u_max_cent = 0.1;    // maximum velocity (located at the centreline)


/* These are "derived" parameters. (Technically, omega is a derived parameter as well, but I'm referring to slightly more complicated derived parameters.) The expression for the viscosity used here
 * is given by kinematic_visc = rho * cs_sq (tau - dt/2), where dt is the timestep. This is Eq. 4.17 in Sec. 4.1.4 of Krüger (Pg. 112); this is a generalisation of Eq. 3.5 in Sec. 3.2.1 of Krüger
 * (Pg. 65), which doesn't include rho. However, we note that an alternate expression for the (nondimensionalised) viscosity is given by Eq. 7.14 in Sec. 7.2.1.1 of Krüger (Pg. 273) as
 * kinematic_visc = cs_sq * (tau - 1/2) * (dx)²/dt. For now, the timestep is 1; putting in an explicit timestepping is one of the outstanding issues identified in
 * couette_wetnode_noneq_BCs_functions.cpp. Putting in an explicit timestepping (dt) and spatial separation (dx) is discussed in Sec. 7.2 of Krüger; this is yet to be implemented both here and in
 * couette_wetnode_noneq_BCs_functions.cpp.
 * 
 * From Sec. 7.2.1.2 and Sec. 7.3.3 in Krüger, we also have the expression for the density drop across the channel (using periodic BCs in the x-direction, and the fact that x = 0 and x = x_len - 1
 * are virtual nodes) as rho_in = rho_out + (dp/dx) * cs_sq * (x_len - 1). We use rho_out = rho_init = 1.0. This is the consequence of Eq. 7.30 and the expressions right after that, provided in Sec.
 * 7.3.3 of Krüger (Pg. 286). */
const double kinemat_visc = rho_start * cs_sq * (tau - 1.0/2.0);                  // kinematic viscosity
const double dpdx = (8.0 * kinemat_visc * u_max_cent)/(std::pow((y_len - 1), 2)); // pressure drop in the x-direction along the lattice
const double rho_in = rho_start + dpdx * cs_sq * (x_len - 1);                     // density in the inlet (using periodicitu in x, and the fact that x = 0 and x = x_len - 1 are virtual nodes)
const double vel_init_sq = std::pow(ux_init_glob, 2) + std::pow(uy_init_glob, 2); // square of the magnitude of the global initial velocity


/* These are the lattice parameters. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
const int q_num = 9; // number of velocity directions (Q in DnQm). Here, Q = 9, with q = 0 as the self-velocity.

const std::vector<double> w = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; // lattice weights

const std::vector<int> cx_int = {0, 1, 0, -1,  0, 1, -1, -1,  1};                        // x-direction lattice speeds
const std::vector<int> cy_int = {0, 0, 1,  0, -1, 1,  1, -1, -1};                        // y-direction lattice speeds
const std::vector<double> cx_float = {0.0, 1.0, 0.0, -1.0,  0.0, 1.0, -1.0, -1.0,  1.0}; // x-direction lattice speeds
const std::vector<double> cy_float = {0.0, 0.0, 1.0,  0.0, -1.0, 1.0,  1.0, -1.0, -1.0}; // y-direction lattice speeds

/* The lattice velocity directions above follow Krüger Pg. 86. These are defined by:
 *
 * index:   0  1  2  3  4  5  6  7  8
 * ----------------------------------
 * x:       0 +1  0 -1  0 +1 -1 -1 +1
 * y:       0  0 +1  0 -1 +1 +1 -1 -1
 *
 * 6 2 5
 *  \|/    ↑ y
 * 3-0-1   |
 *  /|\    --→ x
 * 7 4 8 */


/* This defines the distribution function arrays. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
array3D_float f_coll(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0))); // this is f_i in Krüger Ch. 4
array3D_float f_eq(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));   // this is f_i^eq in Krüger Ch. 4
array3D_float f_prop(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0))); // this is f_i^* in Krüger Ch. 4


/* Here, we define the macroscopic quantities. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
array2D_float rho(x_len, vect1D_float(y_len, rho_start));
array2D_float ux(x_len, vect1D_float(y_len, ux_init_glob));
array2D_float uy(x_len, vect1D_float(y_len, uy_init_glob));

array2D_float vel_sq(x_len, vect1D_float(y_len, vel_init_sq));
array3D_float u_dot_ci(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));


/* Here, we compute the macroscopic quantities (density and the velocity fields) at each time step. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
void compute_macroscopic_params(array3D_float prev_dist_fn, array2D_float &rho, array2D_float &ux, array2D_float &uy) {
  int x_size = prev_dist_fn.size();
  int y_size = prev_dist_fn[0].size();

  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      rho[i][j] = prev_dist_fn[i][j][0] + prev_dist_fn[i][j][1] + prev_dist_fn[i][j][2] + prev_dist_fn[i][j][3] + prev_dist_fn[i][j][4]
                                                                + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][6] + prev_dist_fn[i][j][7] + prev_dist_fn[i][j][8];
      ux[i][j] = (prev_dist_fn[i][j][1] + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][8] - prev_dist_fn[i][j][3] - prev_dist_fn[i][j][6] - prev_dist_fn[i][j][7]) / rho[i][j];
      uy[i][j] = (prev_dist_fn[i][j][2] + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][6] - prev_dist_fn[i][j][4] - prev_dist_fn[i][j][7] - prev_dist_fn[i][j][8]) / rho[i][j];
    }
  }
}

/* Here, we set the initial distribution functions. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
void compute_f_eq(const double sound_speed_sq, array3D_float &f_eq, array2D_float dens, array2D_float x_vel, array2D_float y_vel, array2D_float &vel_sq) {
  int x_num = f_eq.size();
  int y_num = f_eq[0].size();

  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      vel_sq[i][j] = std::pow(x_vel[i][j], 2) + std::pow(y_vel[i][j], 2);

      f_eq[i][j][0] = (4.0/9.0) * dens[i][j] * (1.0 - (vel_sq[i][j]/(2.0 * sound_speed_sq)));
      f_eq[i][j][1] = (1.0/9.0) * dens[i][j] * (1.0 + (x_vel[i][j])/sound_speed_sq + (std::pow(x_vel[i][j], 2))/(2.0 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j])/(2.0 * sound_speed_sq));
      f_eq[i][j][2] = (1.0/9.0) * dens[i][j] * (1.0 + (y_vel[i][j])/sound_speed_sq + (std::pow(y_vel[i][j], 2))/(2.0 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j])/(2.0 * sound_speed_sq));
      f_eq[i][j][3] = (1.0/9.0) * dens[i][j] * (1.0 - (x_vel[i][j])/sound_speed_sq + (std::pow(x_vel[i][j], 2))/(2.0 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j])/(2.0 * sound_speed_sq));
      f_eq[i][j][4] = (1.0/9.0) * dens[i][j] * (1.0 - (y_vel[i][j])/sound_speed_sq + (std::pow(y_vel[i][j], 2))/(2.0 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j])/(2.0 * sound_speed_sq));
      f_eq[i][j][5] = (1.0/36.0) * dens[i][j] * (1.0 + (x_vel[i][j] + y_vel[i][j]) / sound_speed_sq + (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j])/sound_speed_sq);
      f_eq[i][j][6] = (1.0/36.0) * dens[i][j] * (1.0 - (x_vel[i][j] - y_vel[i][j]) / sound_speed_sq - (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j])/sound_speed_sq);
      f_eq[i][j][7] = (1.0/36.0) * dens[i][j] * (1.0 - (x_vel[i][j] + y_vel[i][j]) / sound_speed_sq + (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j])/sound_speed_sq);
      f_eq[i][j][8] = (1.0/36.0) * dens[i][j] * (1.0 + (x_vel[i][j] - y_vel[i][j]) / sound_speed_sq - (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j])/sound_speed_sq);
    }
  }
}

/* Here, we perform the collision step. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
void collision(array3D_float &f_coll, array3D_float previously_propagated_dist_fn, array3D_float current_eq_dist_fn, const double relaxation_time) {
  int x_span = f_coll.size();
  int y_span = f_coll[0].size();
  int q_span = f_coll[0][0].size();

  for (int i = 0; i < x_span; i++) {
    for (int j = 0; j < y_span; j++) {
      for (int k = 0; k < q_span; k++) {
        f_coll[i][j][k] = (1 - (1.0/relaxation_time)) * previously_propagated_dist_fn[i][j][k] + (1.0/relaxation_time) * current_eq_dist_fn[i][j][k];
      }
    }
  }
}

/* Here, we perform the streaming step, without applying the boundary condition at the walls, but with applying periodicity. The details are discussed further in
 * couette_wetnode_noneq_BCs_functions.cpp. */
void streaming_periodic_x_and_y(std::vector<int> discrete_x_vel, std::vector<int> discrete_y_vel, array3D_float &f_prop, array3D_float current_coll_dist_fn) {
  int new_x;
  int new_y;

  int x_range = f_prop.size();
  int y_range = f_prop[0].size();
  int q_range = f_prop[0][0].size();

  for (int i = 0; i < x_range; i++) {
    for (int j = 0; j < y_range; j++) {
      for (int k = 0; k < q_range; k++) {
        new_x = (i + discrete_x_vel[k] + x_range) % x_range;
        new_y = (j + discrete_y_vel[k] + y_range) % y_range;
        f_prop[new_x][new_y][k] = current_coll_dist_fn[i][j][k];
      }
    }
  }
}

/* Here, we apply the macroscopic boundary conditions. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
void macroscopic_BCs_rectangular_channel(array2D_float &rho, array2D_float &ux, array2D_float &uy, array3D_float propagated_dist_fn, const double bottom_wall_vel, const double top_wall_vel) {
  int x_amt = rho.size();
  int y_amt = rho[0].size();

  for (int i = 0; i < x_amt; i++) {
    // Here, we apply the boundary conditions for the bottom wall: continuity, no-slip (ux[i][0] = u_wall_bottom), and no-penetration (uy[i][0] = 0.0).
    rho[i][0] = (1.0/(1.0 - uy[i][0])) *
           (propagated_dist_fn[i][0][0] + propagated_dist_fn[i][0][1] + propagated_dist_fn[i][0][3] + 2.0 * (propagated_dist_fn[i][0][4] + propagated_dist_fn[i][0][7] + propagated_dist_fn[i][0][8]));
    ux[i][0] = bottom_wall_vel;
    uy[i][0] = 0.0;

    // Here, we apply the boundary conditions for the top wall: continuity, no-slip (ux[i][0] = u_wall_top), and no-penetration (uy[i][0] = 0.0).
    rho[i][y_amt - 1] = (1.0/(1.0 + uy[i][y_amt - 1])) * (propagated_dist_fn[i][y_amt - 1][0] + propagated_dist_fn[i][y_amt - 1][1] + propagated_dist_fn[i][y_amt - 1][3]
                                                       + 2.0 * (propagated_dist_fn[i][y_amt - 1][2] + propagated_dist_fn[i][y_amt - 1][5] + propagated_dist_fn[i][y_amt - 1][6]));
    ux[i][y_amt - 1] = top_wall_vel;
    uy[i][y_amt - 1] = 0.0;
  }
}

/* Here, we apply the equilibrium boundary conditions, which provide first-order accuracy. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
void equil_BB_rectangular_channel(const double bottom_wall_velocity, const double top_wall_velocity, const double speed_sound_sq,
                                  vect1D_float vel_weight, vect1D_float x_dir_lattice_vel, array2D_float fluid_density, array3D_float &f_prop) {
  int x_amount = f_prop.size();
  int y_amount = f_prop[0].size();

  for (int i = 0; i < x_amount; i++) {
    // Here, we apply the equilibrium bounce-back condition on the bottom plate.
    f_prop[i][0][2] = f_prop[i][0][4];
    f_prop[i][0][5] = f_prop[i][0][7] + 2.0 * fluid_density[i][0] * vel_weight[5] * (x_dir_lattice_vel[5] * bottom_wall_velocity)/speed_sound_sq;
    f_prop[i][0][6] = f_prop[i][0][8] + 2.0 * fluid_density[i][0] * vel_weight[6] * (x_dir_lattice_vel[6] * bottom_wall_velocity)/speed_sound_sq;

    // Here, we apply the equilibrium bounce-back condition on the top plate, incorporating the wall motion.
    f_prop[i][y_amount - 1][4] = f_prop[i][y_amount - 1][2];
    f_prop[i][y_amount - 1][7] = f_prop[i][y_amount - 1][5] + 2.0 * fluid_density[i][y_amount - 1] * vel_weight[7] * (x_dir_lattice_vel[7] * top_wall_velocity)/speed_sound_sq;
    f_prop[i][y_amount - 1][8] = f_prop[i][y_amount - 1][6] + 2.0 * fluid_density[i][y_amount - 1] * vel_weight[8] * (x_dir_lattice_vel[8] * top_wall_velocity)/speed_sound_sq;
  }
}

/* Here, we apply nonequilibrium boundary conditions, which provide second-order accuracy. Specifically, we apply the nonequilibrium extrapolation method. The details are discussed further in
 * couette_wetnode_noneq_BCs_functions.cpp. */
void NEEM_rectangular_channel(vect1D_float c_x_dir, vect1D_float c_y_dir, array2D_float x_speed, array2D_float y_speed, const double speed_of_sound_squared,
                              array2D_float density, vect1D_float weight, array3D_float equil_dist_fn, array3D_float &f_prop, array3D_float &u_dot_ci) {
  int x_qty = f_prop.size();
  int y_qty = f_prop[0].size();
  int q_qty = f_prop[0][0].size();

  for (int i = 0; i < x_qty; i++) {
    for (int k = 0; k < q_qty; k++) {
      u_dot_ci[i][0][k] = (x_speed[i][0] * c_x_dir[k] + y_speed[i][0] * c_y_dir[k]);
      u_dot_ci[i][y_qty - 1][k] = (x_speed[i][y_qty - 1] * c_x_dir[k] + y_speed[i][y_qty - 1] * c_y_dir[k]);
      f_prop[i][0][k] = weight[k] * (density[i][0] + u_dot_ci[i][0][k]/speed_of_sound_squared) + (f_prop[i][1][k] - equil_dist_fn[i][1][k]);
      f_prop[i][y_qty - 1][k] = weight[k] * (density[i][y_qty - 1] + u_dot_ci[i][y_qty - 1][k]/speed_of_sound_squared) + (f_prop[i][y_qty - 2][k] - equil_dist_fn[i][y_qty - 2][k]);
    }
  }
}

/* Here, we apply nonequilibrium boundary conditions, which provide second-order accuracy. Specifically, we apply the Zou-He (nonequilibrium bounce back) boundary conditions. The details are
 * discussed further in couette_wetnode_noneq_BCs_functions.cpp. */
void Zou_He_rectangular_channel(array2D_float x_dir_velocity, array2D_float y_dir_velocity, array2D_float fluid_dens, array3D_float &f_prop) {
  int x_dir_size = f_prop.size();
  int y_dir_size = f_prop[0].size();

  for (int i = 0; i < x_dir_size; i++) {
    /* Here, we incorporate the Zou-He (nonequilibrium bounce back) method, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48. This first applies the ZH boundary conditions on the
     * bottom plate. */
    f_prop[i][0][2] = f_prop[i][0][4] + (2.0 * fluid_dens[i][0] * y_dir_velocity[i][0])/3.0;
    f_prop[i][0][5] = f_prop[i][0][7] + (fluid_dens[i][0] * y_dir_velocity[i][0])/6.0 - (f_prop[i][0][1] - f_prop[i][0][3])/2.0 + x_dir_velocity[i][0]/2.0;
    f_prop[i][0][6] = f_prop[i][0][8] + (fluid_dens[i][0] * y_dir_velocity[i][0])/6.0 + (f_prop[i][0][1] - f_prop[i][0][3])/2.0 - x_dir_velocity[i][0]/2.0;

    /* Here, we incorporate the Zou-He (nonequilibrium bounce back) method, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48. This now applies the ZH boundary conditions on the top
     * plate. */
    f_prop[i][y_dir_size - 1][4] = f_prop[i][y_dir_size - 1][2] - (2.0 * fluid_dens[i][y_dir_size - 1] * y_dir_velocity[i][y_dir_size - 1])/3.0;
    f_prop[i][y_dir_size - 1][7] = f_prop[i][y_dir_size - 1][5] - (fluid_dens[i][y_dir_size - 1] * y_dir_velocity[i][y_dir_size - 1])/6.0
                                                                + (f_prop[i][y_dir_size - 1][1] - f_prop[i][y_dir_size - 1][3])/2.0 - x_dir_velocity[i][y_dir_size - 1]/2.0;
    f_prop[i][y_dir_size - 1][8] = f_prop[i][y_dir_size - 1][6] - (fluid_dens[i][y_dir_size - 1] * y_dir_velocity[i][y_dir_size - 1])/6.0
                                                                - (f_prop[i][y_dir_size - 1][1] - f_prop[i][y_dir_size - 1][3])/2.0 + x_dir_velocity[i][y_dir_size - 1]/2.0;
  }
}

/* Here, we define the periodic bounce back boundary conditions for the pressure gradient at the inlet at the outlet. We note that x = 0 and x = (x_len - 1) are virtual nodes, where we can apply the
 * pressure conditions. As mentioned above, we use Eq. 7.30 and the expressions right after that, provided in Sec. 7.3.3 of Krüger (Pg. 286). */
void pressure_drop_poiseuille(array3D_float &f_coll, array3D_float current_equil_dist_fn, vect1D_float x_dir_vel_lattice_proj, vect1D_float y_dir_vel_lattice_proj, vect1D_float velocity_weights,
                              array2D_float x_dir_vel_field, array2D_float y_dir_vel_field, const double inlet_density, const double outlet_density, const double speed_of_sound_sq) {
  int x_quantity = f_coll.size();
  int y_quantity = f_coll[0].size();
  int q_quantity = f_coll[0][0].size();

  for (int j = 0; j < y_quantity; j++) {
    for (int k = 0; k < q_quantity; k++) {
      f_coll[0][j][k] = (velocity_weights[k] *
                                 (inlet_density + (x_dir_vel_lattice_proj[k] * x_dir_vel_field[x_quantity - 2][j] + y_dir_vel_lattice_proj[k] * y_dir_vel_field[x_quantity - 2][j])/speed_of_sound_sq))
                        + f_coll[x_quantity - 2][j][k] - current_equil_dist_fn[x_quantity - 2][j][k];
      f_coll[x_quantity - 1][j][k] = (velocity_weights[k] *
                                                          (outlet_density + (x_dir_vel_lattice_proj[k] * x_dir_vel_field[1][j] + y_dir_vel_lattice_proj[k] * y_dir_vel_field[1][j])/speed_of_sound_sq))
                                     + f_coll[2][j][k] - current_equil_dist_fn[2][j][k];
    }
  }
}

/* Here, we define the function that prints a rank-2 matrix. */
void print_2D_array(array2D_float input_mat) {
  int num_cols = input_mat.size();
  int num_rows = input_mat[0].size();

  for (int i = 0; i < num_cols; i++) {
    std::cout << "[";
    for (int j = 0; j < num_rows - 1; j++) {
      std::cout << input_mat[i][j] << " ";
    }
    std::cout << input_mat[i][num_rows - 1] << "]" << std::endl;
  }
}


////////// THE MAIN SIMULATION LOOP STARTS HERE //////////

int main() {

  ////////// THE INITIALISATION STARTS HERE //////////

  /* Here, we initialise the equilibrium distribution function f_eq, as well as the distribution functions used in the collision step (f_coll) and the streaming step (f_prop). The details are
   * discussed further in couette_wetnode_noneq_BCs_functions.cpp. This uses the compute_f_eq function defined on line 126. */
  compute_f_eq(cs_sq, f_eq, rho, ux, uy, vel_sq);
  f_coll = f_eq;
  f_prop = f_eq;


  ////////// THE MAIN SIMULATION LOOP STARTS HERE //////////

  for (int t = 0; t < t_steps; t++) {

    /* Here, we perform the main simulation loop. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. We continue with the ordering given there; namely, the order given by:
     *
     *            calculating macroscopic parameters -> imposing macroscopic BCs -> imposing noneq BCs -> computing f_eq and colliding -> imposing 1st order BCs -> streaming -> repeat
     *
     * We start with calculating the macroscopic parameters. This uses the compute_macroscopic_params function defined on line 111. */
    compute_macroscopic_params(f_prop, rho, ux, uy);


    /* Here, we apply the macroscopic boundary conditions. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. This uses the macroscopic_BCs_rectangular_channel function
     * defined on line 111. */
    macroscopic_BCs_rectangular_channel(rho, ux, uy, f_prop, u_wall_bottom, u_wall_top);


    /* Here, we compute the equilibrium distribution at the given timestep. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. This uses the compute_f_eq function defined on
     * line 126. */
    compute_f_eq(cs_sq, f_eq, rho, ux, uy, vel_sq);


    /* Here, we perform the collision step. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. This uses the collision function defined on line 148. */
    collision(f_coll, f_prop, f_eq, tau);


    /* Here, we apply the periodic bounce back boundary conditions, with x = 0 and x = (x_len - 1) as virtual nodes. As discussed above (on lines 55-64), from Sec. 7.2.1.2 and Sec. 7.3.3 in Krüger, we
     * have the expression for the density drop across the channel (using periodic BCs in the x-direction, and the fact that x = 0 and x = (x_len - 1)) are virtual nodes) as
     * rho_in = rho_out + (dp/dx) * cs_sq * (x_len - 1). We use rho_out = rho_init = 1.0. This is the consequence of Eq. 7.30 and the expressions right after that, provided in Sec. 7.3.3 of Krüger (Pg.
     * 286). This uses the pressure_drop_poiseuille function defined on line 265. */
    pressure_drop_poiseuille(f_coll, f_eq, cx_float, cy_float, w, ux, uy, rho_in, rho_start, cs_sq);


    /* Here, we perform the streaming step, without applying the boundary condition at the wall, but with applying the periodic boundary conditions. The details are discussed further in
     * couette_wetnode_noneq_BCs_functions.cpp. This uses the streaming_periodic_x_and_y function defined on line 164. */
    streaming_periodic_x_and_y(cx_int, cy_int, f_prop, f_coll);


    /* Here, we apply the equilibrium bounce-back conditions on the top and the bottom. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. This uses the
     * equil_BB_rectangular_channel function defined on line 164. */
    equil_BB_rectangular_channel(u_wall_bottom, u_wall_top, cs_sq, w, cx_float, rho, f_prop);


    /* Here, we apply the nonequilibrium boundary conditions. The details are discussed further in couette_wetnode_noneq_BCs_functions.cpp. We can use either the nonequilibrium extrapolation method
     * (given here by NEEM_rectangular_channel, defined on line 224) or the Zou-He function (given here by Zou_He_rectangular_channel, defined on line 242). Both are presented here. */
     // NEEM_rectangular_channel(cx_float, cy_float, ux, uy, cs_sq, rho, w, f_eq, f_prop, u_dot_ci);
    Zou_He_rectangular_channel(ux, uy, rho, f_prop);
  }

  /* Here, we output the velocities and the densities, to test against the analytic solution. The problem has infinite symmetry in the x-direction, and the velocity depends *only* on the profile in
   * the y-direction. In particular, the analytic solution should be ux = ux(y) = (dp/dx) * ((y_len)²-y²) * (-1/2μ), where y is a variable, uy = 0, and rho = 1 (since this is an incompressible
   * problem. This uses the print_2D_array function defined on line 284. */
   // This outputs the velocity in the x-direction.
  print_2D_array(ux);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  // This outputs the density.
  print_2D_array(rho);

}