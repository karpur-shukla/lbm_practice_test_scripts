/* As a first step, we use the lattice Boltzmann (LB) technique to simulate the problem of 2D Couette flow in rectangular coordinates. (This is the problem of viscous flow between two rectangular
 * plates with no pressure drop, where the top wall moves at a given velocity and the bottom wall is stationary. We use no-slip boundary conditions at the walls, and assume a pressure gradient of
 * zero.) For a channel of length L and a top wall velocity of U0, the velocity is given by u = U0 * y / L.
 *
 * Here, we heavily reference Krüger et al., The Lattice Boltzmann Method, Springer Nature, Cham, Switzerland, 2017.
 *
 * For the LB simulation, we use the following setup:
 *   Collision operator:    BGK approximation
 *   Lattice structure:     D2Q9
 *   1st order BCs:         Periodic in x; wet-node bounce-back (c.f. Krüger Pg. 177) in y at walls
 *   2nd order (noneq) BCs: Zou-He (nonequilibrium bounce back) (c.f. Krüger Pgs. 196-199). The equations for the nonequilibrium extrapolation method (c.f. Krüger Pgs. 194-195) are also included for
 *                          reference.
 *   Relaxation scheme:     Single relaxation time
 *   Added forces:          None
 *   Number of phases:      1
 *
 * Note that for extensions to any meaningful problem, we WILL need to consider the minefield of noneq. BCs (e.g., Zou-He type, Inamuro type), extend the collision operator beyond BGK
 * (e.g., Shakhov), consider multiple relaxation times, consider multiphase, and eventually move to finite-volume LBM/DUGKS.
 *
 * CURRENT ISSUES WITH THIS THAT NEED TO BE SOLVED:
 * - NEED TO FIGURE OUT HOW TO CALCULATE VISCOSITY. NOT SURE HOW TO RESOLVE EQ. 3.5 IN KRÜGER (PG. 65) WITH THE FACT THAT I'M USING TIMESTEPS AS AN INPUT VARIABLE WHICH PROVIDES GREATER RESOLUTION.
 *   VISCOSITY CAN *NOT* BE RESOLUTION-DEPENDENT!
 *   -- UPDATE AT LINES 55-69.
 * - NEED TO INCORPORATE THE EXPLICIT TIMESTEPPING IN THE EXPRESSION FOR VISCOSITY (KRÜGER EQ. 3.5, PG. 66); IN THE UPDATE FOR THE DISTRIBUTION FUNCTION (KRÜGER EQ. 3.8 & 3.9, PG. 66; KRÜGER EQ.
     3.13, PG. 70), AND THE BOUNDARY CONDITIONS
 * - NEED TO UNDERSTAND WHERE SPEED OF SOUND IN LATTICE UNITS ACTUALLY COMES FROM. RIGHT NOW, IT'S JUST SET TO 1/3.
     -- NOTE THAT CS = C/3, WHERE C = DX/DT. */


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


/* Here, we define the 2D array class and 3D array class as specific types. This is to simplify the expressions when we initialise the 1-particle probability distribution functions, which are 3D
 * arrays. With this, we can avoid writing std::vectors inside other std::vectors, which quickly becomes impossible to read or deal with. For reference, before I did the typedef, one of these looked
 * like: std::vector<std::vector<std::vector<double>>> f(x_len, std::vector<std::vector<double>>(y_len, std::vector<double>(q_num))). */
typedef std::vector<double> vect1D_float;
typedef std::vector<std::vector<double>> array2D_float;
typedef std::vector<std::vector<std::vector<double>>> array3D_float;


/* These are the simulation parameters, including some "extra" simulation parameters. Following the suggestion in Krüger Sec. 3.3.3.4 (Pgs. 69 - 70), I'm explicitly defining cs_sq and cs_four (speed
 * of sound squared and speed of sound to the fourth). Apparently, calling these is faster than calculating cs^2 and cs^4 in the loops. For the initialisation, I also define vel_init_sq (the square
 * of the TOTAL initial velocity; i.e. (ux_init)^2 + (uy_init)^2). Defining vel_init_sq just lets the initialisation be more comprehensible.*/
const int x_len = 40;                   // number of x grid points. System is periodic in x.
const int y_len = 20;                   // number of y grid points
const int t_steps = 10001;           // number of time steps. Since I update the densities and velocities at the beginning of the simulation, I'm doing one extra timestep.
const double tau = 0.9;                 // BGK relaxation time
const double omega = 1.0/tau;           // BGK inverse relaxation time (lowercase omega, NOT capital omega)
const double rho_init = 1.0;            // initial density
const double cs_sq = 1.0/3.0;           // speed of sound squared in lattice units
const double u_wall_top = 0.1;          // top wall velocity
const double u_wall_bottom = 0.0;       // bottom wall velocity
const double ux_init = 0.0;             // global initial velocity in the x-direction
const double uy_init = 0.0;             // global initial velocity in the y-direction

const double vel_init_sq = std::pow(ux_init, 2) + std::pow(uy_init, 2); // square of the magnitude of the initial velocity

/* These are the lattice parameters. We define the x-direction and y-direction lattice speeds as both a float vector and an int vector. The float vector goes into equations in which the lattice
 * projections (the c_i's, i.e., the x-direction and y-direction lattice speeds) are used. (Usually, this involves expressions involving the dot product of u and c.) */
const int q_num = 9; // number of velocity directions (Q in DnQm). Here, Q = 9, with q = 0 as self-velocity.

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


 /* This defines the distribution function arrays. Since the distribution functions live in phase space, we have one f for each value of x, y, AND q; i.e., at each node, we have q_num different
  * distribution functions. (Here, q_num = 9.) Thus, if we want to consider each f, f_eq, and f_prop (the distribution function, equilibrium distribution function, and propagated distribution
  * function respectively), we need to make three rank-3 arrays. */
array3D_float f_coll(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0))); // this is f_i in Krüger Ch. 4
array3D_float f_eq(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));   // this is f_i^eq in Krüger Ch. 4
array3D_float f_prop(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0))); // this is f_i^* in Krüger Ch. 4


/* Here, we define the macroscopic quantities (for this problem, the density and velocity fields). In their declaration, we initialise them up-front with their initial values (rho_init for the
 * initial density, and ux_init and uy_init respectively for the initial x- and y-velocities). This will allow us to immediately initialise the equilibrium distribution function f_eq using the exact
 * same compute_f_eq function that we use to compute f_eq at each timestep. */
array2D_float rho(x_len, vect1D_float(y_len, rho_init));
array2D_float ux(x_len, vect1D_float(y_len, ux_init));
array2D_float uy(x_len, vect1D_float(y_len, uy_init));

array2D_float vel_sq(x_len, vect1D_float(y_len, vel_init_sq));
array3D_float u_dot_ci(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));


/* Here, we compute the macroscopic quantities (density and the velocity fields) at each time step. Following the suggestion in Krüger (Sec. 3.3.3.3, Pg. 69), we unroll the calculations in momentum
 * space and perform each calculation explicitly. I have a good handle on how the loop in momentum space would be input here; I'm just doing this at Krüger's suggestion. These expressions are taken
 * from the aforementioned section in Krüger (specifically, on Pg. 69, Eq. 3.12). For completeness, we also show what the expression in the k loop would look like. Since we compute these expressions
 * first, we calculate them using the distribution function streamed from the *previous* time step. */
void compute_macroscopic_params(int x_size, int y_size, array3D_float prev_dist_fn, array2D_float &rho, array2D_float &ux, array2D_float &uy) {
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      rho[i][j] = prev_dist_fn[i][j][0] + prev_dist_fn[i][j][1] + prev_dist_fn[i][j][2] + prev_dist_fn[i][j][3] + prev_dist_fn[i][j][4]
                                                                + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][6] + prev_dist_fn[i][j][7] + prev_dist_fn[i][j][8];
      ux[i][j] = (prev_dist_fn[i][j][1] + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][8] - prev_dist_fn[i][j][3] - prev_dist_fn[i][j][6] - prev_dist_fn[i][j][7]) / rho[i][j];
      uy[i][j] = (prev_dist_fn[i][j][2] + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][6] - prev_dist_fn[i][j][4] - prev_dist_fn[i][j][7] - prev_dist_fn[i][j][8]) / rho[i][j];

//      sum_rho = 0.0;
//      for (int k = 0; k < q_num; k++) {
//        sum_rho += dist_fn[i][j][k];
//        rho[i][j] = sum_rho;
//        ux[i][j] = (prev_dist_fn[i][j][1] + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][8] - prev_dist_fn[i][j][3] - prev_dist_fn[i][j][6] - prev_dist_fn[i][j][7]) / rho[i][j];
//        uy[i][j] = (prev_dist_fn[i][j][2] + prev_dist_fn[i][j][5] + prev_dist_fn[i][j][6] - prev_dist_fn[i][j][4] - prev_dist_fn[i][j][7] - prev_dist_fn[i][j][8]) / rho[i][j];
//      }
    }
  }
}

/* Here, we set the initial distribution functions. Following the suggestion in Krüger Sec. 3.4.7.5 (Pgs. 92-93), I'm "unrolling" the loop over momentum space, and calculating f_colli][j][k = 1] to
 * f_coll[i][j][k = 9] explicitly. According to Krüger, this makes the calculations faster. These also seem to be more accurate for some reason. These equations are given by expanding Eq. 3.54 (Pg.
 * 82) in Krüger (Sec. 3.4.5). These can be further simplified to the equations given in the aforementioned section in Krüger (specifically, on Pg. 93, Eq. 3.65). For completeness, we also show what
 * the "rolled" expression (using the loop in momentum space) would look like.
 *
 * We note that this is used *both* for the initialisation *and* inside each timestep. We use this up front to initialise the equilibrium distribution function, using the pre-initialised x-velocity,
 * y-velocity, and density arrays. (These were initialised when they were declared already.) The speed of sound is a constant in this setup. */
void compute_f_eq(int x_num, int y_num, double sound_speed_sq, array3D_float &f_eq, array2D_float dens, array2D_float x_vel, array2D_float y_vel, array2D_float &vel_sq) {
  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      vel_sq[i][j] = std::pow(ux[i][j], 2) + std::pow(uy[i][j], 2);

      f_eq[i][j][0] = (4.0 / 9.0) * dens[i][j] * (1.0 - (vel_sq[i][j] / (2 * sound_speed_sq)));
      f_eq[i][j][1] = (1.0 / 9.0) * dens[i][j] * (1.0 + (x_vel[i][j]) / sound_speed_sq + (std::pow(x_vel[i][j], 2)) / (2 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j]) / (2 * sound_speed_sq));
      f_eq[i][j][2] = (1.0 / 9.0) * dens[i][j] * (1.0 + (y_vel[i][j]) / sound_speed_sq + (std::pow(uy[i][j], 2)) / (2 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j]) / (2 * sound_speed_sq));
      f_eq[i][j][3] = (1.0 / 9.0) * dens[i][j] * (1.0 - (x_vel[i][j]) / sound_speed_sq + (std::pow(x_vel[i][j], 2)) / (2 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j]) / (2 * sound_speed_sq));
      f_eq[i][j][4] = (1.0 / 9.0) * dens[i][j] * (1.0 - (y_vel[i][j]) / sound_speed_sq + (std::pow(uy[i][j], 2)) / (2 * std::pow(sound_speed_sq, 2)) - (vel_sq[i][j]) / (2 * sound_speed_sq));
      f_eq[i][j][5] = (1.0 / 36.0) * dens[i][j] * (1.0 + (x_vel[i][j] + y_vel[i][j]) / sound_speed_sq + (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j]) / sound_speed_sq);
      f_eq[i][j][6] = (1.0 / 36.0) * dens[i][j] * (1.0 - (x_vel[i][j] - y_vel[i][j]) / sound_speed_sq - (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j]) / sound_speed_sq);
      f_eq[i][j][7] = (1.0 / 36.0) * dens[i][j] * (1.0 - (x_vel[i][j] + y_vel[i][j]) / sound_speed_sq + (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j]) / sound_speed_sq);
      f_eq[i][j][8] = (1.0 / 36.0) * dens[i][j] * (1.0 + (x_vel[i][j] - y_vel[i][j]) / sound_speed_sq - (x_vel[i][j] * y_vel[i][j]) / (std::pow(sound_speed_sq, 2)) + (vel_sq[i][j]) / sound_speed_sq);

//    Note that to use this loop, we need to add four extra arguments to this function: vect1D_float weight = w, vect1D_float &c_x_dir = cx_float, vect1D_float &c_y_dir = cy_float, and
//    array3D_float &u_dot_ci.
//      for (int k = 0; k < q_num; k++) {
//        u_dot_ci[i][j][k] = (x_vel[i][j] * cx_float[k] + y_vel[i][j] * cy_float[k]);
//        f_eq[i][j][k] = w[k] * dens[i][j] * (1 + u_dot_ci[i][j][k] / sound_speed_sq + std::pow(u_dot_ci[i][j][k], 2) / (2 * std::pow(sound_speed_sq, 2)) - vel_sq[i][j] / (2 * sound_speed_sq));
//      }
    }
  }
}

/* Here, we perform the collision step. This uses the simplified expression, Eq. 3.13, in Krüger Sec. 3.3.3.5 (Pg. 70). THIS DOES NOT YET INCLUDE A VARIABLE TIMESTEP PARAMETER. Across the collision,
 * streaming, and boundary condition steps, the collision and propagated distribution functions (f_coll and f_prop respectively here, f_i and f_i^* in Krüger Ch. 4) swap places (during the collision
 * step) and then swap places again (during the streaming step.) Here, we collide (update f_coll). We will then stream (update f_prop) and then apply boundary conditions to f_prop. */
void collision(int x_span, int y_span, int q_span, array3D_float &f_coll, array3D_float previously_propagated_dist_fn, array3D_float current_eq_dist_fn, double relaxation_time) {
  for (int i = 0; i < x_span; i++) {
    for (int j = 0; j < y_span; j++) {
      for (int k = 0; k < q_span; k++) {
        f_coll[i][j][k] = (1 - (1.0 / relaxation_time)) * previously_propagated_dist_fn[i][j][k] + (1.0 / relaxation_time) * current_eq_dist_fn[i][j][k];
      }
    }
  }
}

/* Here, we perform the streaming step, without applying the boundary condition at the wall, but with applying the periodic boundary conditions. The boundary condition at the wall shifts everything
 * over, and this will be applied in a different function. */
void streaming_periodic_x_and_y(int x_range, int y_range, int q_range, std::vector<int> discrete_x_vel, std::vector<int> discrete_y_vel, array3D_float &f_prop, array3D_float current_coll_dist_fn) {
  int new_x;
  int new_y;

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

/* Here, we apply the macroscopic boundary conditions. As mentioned in Krüger Sec. 5.3.4.1 (Pg. 190), since the lattice Boltzmann equation doesn't directly deal with the macroscopic fields (density
 * and velocity), we need to explicitly impose the continuity equation, the no-slip condition, and the no-penetration condition.
 * 
 * We note that the macroscopic boundary conditions need to be defined *per the specific problem*, since we need to satisfy this per the *given* geometry and boundary conditions. Thus, we can't
 * define a *general* function to impose the macroscopic boundary conditions (unless we're doing something fancy like recreating Palabos); instead, we need to define the macroscopic boundary
 * condition function *per problem*.
 
 * The wall density is straightforwardly calculated in the aforementioned section of Krüger. However, we note that there is an explicit de-linking of timestep and space-step in this derivation. Thus,
 * for a speed c = dx/dt for timestep dt and space step dx, expressions that should be c/(c ± u_wall) are instead written here as 1/(1 ± u_wall). This uses Eq. 5.32 and Eq. 5.33 (Pg. 191) in
 * Krüger. */
void macroscopic_BCs_Couette(int x_amt, int y_amt, array2D_float &rho, array2D_float &ux, array2D_float &uy, array3D_float propagated_dist_fn, double bottom_wall_vel, double top_wall_vel) {
  for (int i = 0; i < x_amt; i++) {
    // Here, we apply the boundary conditions for the bottom wall: continuity, no-slip (ux[i][0] = u_wall_bottom), and no-penetration (uy[i][0] = 0.0).
    rho[i][0] = (1.0 / (1.0 - uy[i][0])) *
           (propagated_dist_fn[i][0][0] + propagated_dist_fn[i][0][1] + propagated_dist_fn[i][0][3] + 2.0 * (propagated_dist_fn[i][0][4] + propagated_dist_fn[i][0][7] + propagated_dist_fn[i][0][8]));
    ux[i][0] = bottom_wall_vel;
    uy[i][0] = 0.0;

    // Here, we apply the boundary conditions for the top wall: continuity, no-slip (ux[i][0] = u_wall_top), and no-penetration (uy[i][0] = 0.0).
    rho[i][0] = (1.0 / (1.0 + uy[i][y_amt - 1])) * (propagated_dist_fn[i][y_amt - 1][0] + propagated_dist_fn[i][y_amt - 1][1] + propagated_dist_fn[i][y_amt - 1][3]
                    + 2.0 * (propagated_dist_fn[i][y_amt - 1][2] + propagated_dist_fn[i][y_amt - 1][5] + propagated_dist_fn[i][y_amt - 1][6]));
    ux[i][y_amt - 1] = top_wall_vel;
    uy[i][y_amt - 1] = 0.0;
  }
}

/* Here, we apply the equilibrium boundary conditions onto the newly-streamed f_prop, which provide first-order accuracy. This uses the bounce back scheme, modified for the wet-node layout (as was
 * done here). Here, it's easier to use the explicit expressions given in Eq. 5.25, 5.27, and 5.28 in Krüger, rather than the general expression Eq. 5.26 in Krüger. Like with the "unrolled"
 * expressions for rho, u_x, and u_y; this is not done because implementing those expressions is particularly challenging. */
void equil_BB_Couette(int x_amount, int y_amount, double bottom_wall_velocity, double top_wall_velocity, double speed_sound_sq,
                       vect1D_float vel_weight, vect1D_float x_dir_lattice_vel, array2D_float fluid_density, array3D_float &f_prop) {
  for (int i = 0; i < x_amount; i++) {
    // Here, we apply the equilibrium bounce-back condition on the bottom plate.
    f_prop[i][0][2] = f_prop[i][0][4];
    f_prop[i][0][5] = f_prop[i][0][7] + 2 * fluid_density[i][0] * vel_weight[5] * (x_dir_lattice_vel[5] * bottom_wall_velocity) / speed_sound_sq;
    f_prop[i][0][6] = f_prop[i][0][8] + 2 * fluid_density[i][0] * vel_weight[6] * (x_dir_lattice_vel[6] * bottom_wall_velocity) / speed_sound_sq;

    // Here, we apply the equilibrium bounce-back condition on the top plate, incorporating the wall motion.
    f_prop[i][y_amount - 1][4] = f_prop[i][y_amount - 1][2];
    f_prop[i][y_amount - 1][7] = f_prop[i][y_amount - 1][5] + 2 * fluid_density[i][y_amount - 1] * vel_weight[7] * (x_dir_lattice_vel[7] * top_wall_velocity) / speed_sound_sq;
    f_prop[i][y_amount - 1][8] = f_prop[i][y_amount - 1][6] + 2 * fluid_density[i][y_amount - 1] * vel_weight[8] * (x_dir_lattice_vel[8] * top_wall_velocity) / speed_sound_sq;
  }
}

/* Here, we apply nonequilibrium boundary conditions onto the newly-streamed f_prop, which provide second-order accuracy. This function in particular uses the nonequilibrium extrapolation method
 * (NEEM), discussed in Krüger Sec. 5.3.4.3, Pgs. 194-196, Eq. 5.40-5.41. (Those equations are originally derived from Eq. Sec. 4.2.4 in Krüger, Pgs. 118-119.) As with the macroscopic boundary
 * conditions, we need to define the NEEM boundary conditions *per the specific problem*, since we need to satisfy this per the *given* geometry and boundary conditions. Thus, we can't define a
 * *general* function to impose the NEEM boundary conditions (unless we're doing something fancy like recreating Palabos); instead, we need to define the NEEM boundary condition function *per
 * problem.* This uses Eq. 5.40-5.41 (Pgs. 194-196) in Krüger. THIS DOES NOT YET INCORPORATE THE MULTIPLICATIVE FACTOR OF TAU. 
 * 
 * Surprisingly enough, the values of this change slightly depending on whether cx_float and cy_float are initialised as float vectors or double vectors! */
void NEEM_Couette(int x_qty, int y_qty, int q_qty, vect1D_float c_x_dir, vect1D_float c_y_dir, array2D_float x_speed, array2D_float y_speed, double speed_of_sound_squared, array2D_float density, 
                  vect1D_float weight, array3D_float equil_dist_fn, array3D_float &f_prop, array3D_float &u_dot_ci) {
  for (int i = 0; i < x_len; i++) {
    for (int k = 0; k < q_num; k++) {
      u_dot_ci[i][0][k] = (x_speed[i][0] * c_x_dir[k] + y_speed[i][0] * c_y_dir[k]);
      u_dot_ci[i][y_qty - 1][k] = (x_speed[i][y_qty - 1] * c_x_dir[k] + y_speed[i][y_qty - 1] * c_y_dir[k]);
      f_prop[i][0][k] = weight[k] * (density[i][0] + u_dot_ci[i][0][k] / speed_of_sound_squared) + (f_prop[i][1][k] - equil_dist_fn[i][1][k]);
      f_prop[i][y_qty - 1][k] = weight[k] * (density[i][y_len - 1] + u_dot_ci[i][y_qty - 1][k] / speed_of_sound_squared) + (f_prop[i][y_qty - 2][k] - equil_dist_fn[i][y_qty - 2][k]);
    }
  }
}

/* Here, we apply nonequilibrium boundary conditions onto the newly-streamed f_prop, which provide second-order accuracy. This function in particular uses the non-equilibrium (second-order)
 * expressions in the wet-node layout. Here, specifically, we use the Zou-He (nonequilibrium bounce back) boundary conditions, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199. As with the macroscopic
 * boundary conditions, we need to define the Zou-He boundary conditions *per the specific problem*, since we need to satisfy this per the *given* geometry and boundary conditions. Thus, we can't
 * define a *general* function to impose the Zou-He boundary conditions (unless we're doing something fancy like recreating Palabos); instead, we need to define the Zou-He boundary condition function
 * *per problem.* Note that, in particular, the Zou-He boundary conditions involve solving a specific set of linear equations at the boundaries. This function uses Eq. 5.42-5.48 (Pgs. 196-199) in\
 * Krüger. */
void Zou_He_Couette(int x_dir_size, int y_dir_size, array2D_float x_dir_velocity, array2D_float y_dir_velocity, array3D_float &f_prop) {
  for (int i = 0; i < x_len; i++) {

    /* Here, we incorporate the Zou-He (nonequilibrium bounce back) method, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48. This first applies the ZH boundary conditions on the
     * bottom plate. */
    f_prop[i][0][2] = f_prop[i][0][4] + (2.0 * y_dir_velocity[i][0])/3.0;
    f_prop[i][0][5] = f_prop[i][0][7] + y_dir_velocity[i][0]/6.0 - (f_prop[i][0][1] - f_prop[i][0][3])/2.0 + x_dir_velocity[i][0]/2.0;
    f_prop[i][0][6] = f_prop[i][0][8] + y_dir_velocity[i][0]/6.0 + (f_prop[i][0][1] - f_prop[i][0][3])/2.0 - x_dir_velocity[i][0]/2.0;

    /* Here, we incorporate the Zou-He (nonequilibrium bounce back) method, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48. This now applies the ZH boundary conditions on the top
     * plate. */
    f_prop[i][y_len - 1][4] = f_prop[i][y_dir_size - 1][2] - (2.0 * y_dir_velocity[i][y_dir_size - 1])/3.0;
    f_prop[i][y_len - 1][7] = f_prop[i][y_dir_size - 1][5] - y_dir_velocity[i][y_dir_size - 1]/6.0
                                                           + (f_prop[i][y_dir_size - 1][1] - f_prop[i][y_dir_size - 1][3])/2.0 - x_dir_velocity[i][y_dir_size - 1]/2.0;
    f_prop[i][y_len - 1][8] = f_prop[i][y_dir_size - 1][6] - y_dir_velocity[i][y_dir_size - 1]/6.0
                                                           - (f_prop[i][y_dir_size - 1][1] - f_prop[i][y_dir_size - 1][3])/2.0 + x_dir_velocity[i][y_dir_size - 1]/2.0;
  }
}

/* Here, we define the function that prints a rank-2 matrix. For now, this will simply be a function that relies on the input x-length and y-length; however, I intend to turn this into a function
 * that prints *any* rank-2 matrix, regardless of shape or contents (i.e., it can print array2D_floats, std::vector<std::vector<int>>s, etc.), and without knowing the dimension ahead of time. */
void print_2D_array(int num_cols, int num_rows, array2D_float input_mat) {
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

  ////////// THE INITIALISATION iS HERE //////////

  /* Here, we initialise the equilibrium distribution function f_eq, as well as the distribution functions used in the collision step (f_coll) and the streaming step (f_prop). The process of
   * initialising the equilibrium distribution function is the same as the process of updating it at each timestep, using properly initialised expressions for the density and x- and y-velocity
   * arrays. These were properly initialised up front (when they were declared, on lines 103-105), so we can use the same compute_f_eq function as we do to compute f_eq at each timestep. The vel_sq
   * array may or may not be initialised properly, but since compute_f_eq updates vel_sq up front, it doesn't matter. This uses the compute_f_eq function defined on line 141. */
  compute_f_eq(x_len, y_len, cs_sq, f_eq, rho, ux, uy, vel_sq);
  f_coll = f_eq;
  f_prop = f_eq;


  ////////// THE MAIN SIMULATION LOOP STARTS HERE //////////
  for (int t = 0; t < t_steps; t++) {
  
    // Here, we compute the macroscopic quantities (density and the velocity fields) at each time step. This uses the compute_macroscopic_params function defined on line 115.
    compute_macroscopic_params(x_len, y_len, f_prop, rho, ux, uy);


    /* Here, we compute the equilibrium distribution at the given timestep. As before, this explicitly unrolls the loop over momentum space in the calculation of the distribution functions, by
     * expanding Eq. 3.54 (Pg. 82) in Krüger (Sec. 3.4.5). These can, as always, be further simplified to the equations given in Eq. 3.65 (Pg. 93) in Krüger. This uses the compute_f_eq function
     * defined on line 141.*/
    compute_f_eq(x_len, y_len, cs_sq, f_eq, rho, ux, uy, vel_sq);


    /* Here, we perform the collision step. THIS DOES NOT YET INCLUDE A VARIABLE TIMESTEP PARAMETER. Across the collision, streaming, and boundary condition steps, the collision and propagated
     * distribution functions (f_coll and f_prop respectively here, f_i and f_i^* in Krüger Ch. 4) swap places (during the collision step) and then swap places again (during the streaming step.)
     * Here, we collide (update f_coll). We will then stream (update f_prop) and then apply boundary conditions to f_prop. This uses the collision function defined on line 167. */
    collision(x_len, y_len, q_num, f_coll, f_prop, f_eq, tau);


    /* Here, we perform the streaming step, without applying the boundary condition at the wall, but with applying the periodic boundary conditions. The boundary condition at the wall shifts
     * everything over, and this will be applied momentarily. This uses the streaming_periodic_x_and_y function defined on line 180. */
    streaming_periodic_x_and_y(x_len, y_len, q_num, cx_int, cy_int, f_prop, f_coll);


    /* Here, we apply the macroscopic boundary conditions. As mentioned in Krüger Sec. 5.3.4.1 (Pg. 190), since the lattice Boltzmann equation doesn't directly deal with the macroscopic fields
     * (density and velocity), we need to explicitly impose the continuity equation, the no-slip condition, and the no-penetration condition.
     * 
     * We note that the macroscopic boundary conditions need to be defined *per the specific problem*, since we need to satisfy this per the *given* geometry and boundary conditions. Thus, we can't
     * define a *general* function to impose the macroscopic boundary conditions (unless we're doing something fancy like recreating Palabos); instead, we need to define this function *per problem*.
     * This uses the macroscopic_BCs_Couette function defined on line 204.*/
    macroscopic_BCs_Couette(x_len, y_len, rho, ux, uy, f_prop, u_wall_bottom, u_wall_top);


    /* Here, we apply the boundary conditions onto the newly-streamed f_prop. The equilibrium boundary conditions provide first-order accuracy, whereas the nonequilibrium boundary conditions provide
     * second-order accuracy. As with the macroscopic boundary conditions, we need to define the boundary conditions *per the specific problem*, since we need to satisfy this per the *given* geometry
     * and boundary conditions. Thus, we can't define a *general* function to impose the boundary conditions; instead, we need to define the boundary condition function *per problem.* We can use
     * either the equilibrium bounce-back function (given here by the equilibrium bounce-back function equil_BB_Couette, defined on line 225), the nonequilibrium extrapolation method (given here by 
     * NEEM_Couette, defined on line 247) or the Zou-He function (given here by Zou_He_Couette, defined on line 265). All three are presented here, and since Couette flow has a linear profile, they
     * all give the same results. My personal preference is for the Zou-He method, because it's the most intuitive to me. */
    equil_BB_Couette(x_len, y_len, u_wall_bottom, u_wall_top, cs_sq, w, cx_float, rho, f_prop);
//    NEEM_Couette(x_len, y_len, q_num, cx_float, cy_float, ux, uy, cs_sq, rho, w, f_eq, f_prop, u_dot_ci);
//    Zou_He_Couette(x_len, y_len, ux, uy, f_prop);
  }


  /* Here, we output the velocities and the densities, to test against the analytic solution. The problem has infinite symmetry in the x-direction, and the velocity depends *only* on the profile in
   * the y-direction. In particular, the analytic solution should be ux = ux(y) = ux_init * y / y_len (where y is a variable), uy = 0, and rho = 1 (since this is an incompressible problem). */
  // Outputting ux.
  print_2D_array(x_len, y_len, ux);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  // Outputting uy.
  print_2D_array(x_len, y_len, uy);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  // Outputting density.
  print_2D_array(x_len, y_len, rho);


}