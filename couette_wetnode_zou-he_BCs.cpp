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

const std::vector<float> w = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; // lattice weights

const std::vector<int> cx_int = {0, 1, 0, -1,  0, 1, -1, -1,  1};                       // x-direction lattice speeds
const std::vector<int> cy_int = {0, 0, 1,  0, -1, 1,  1, -1, -1};                       // y-direction lattice speeds
const std::vector<float> cx_float = {0.0, 1.0, 0.0, -1.0,  0.0, 1.0, -1.0, -1.0,  1.0}; // x-direction lattice speeds
const std::vector<float> cy_float = {0.0, 0.0, 1.0,  0.0, -1.0, 1.0,  1.0, -1.0, -1.0}; // y-direction lattice speeds

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
  * function), we need to make three rank-3 arrays. */
array3D_float f(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));      // this is f_i in Krüger Ch. 4
array3D_float f_eq(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));   // this is f_i^eq in Krüger Ch. 4
array3D_float f_prop(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0))); // this is f_i^* in Krüger Ch. 4


// Here, we define the macroscopic quantities (for this problem, the density and velocity fields).
array2D_float rho(x_len, vect1D_float(y_len, rho_init));
array2D_float ux(x_len, vect1D_float(y_len, ux_init));
array2D_float uy(x_len, vect1D_float(y_len, uy_init));

array2D_float vel_sq(x_len, vect1D_float(y_len, vel_init_sq));
array3D_float u_dot_ci(x_len, array2D_float(y_len, vect1D_float(q_num, 0.0)));


////////// THE MAIN SIMULATION LOOP STARTS HERE //////////

// Right now, the entire LB simulation (initialisation included) is done in main(). This will eventually be broken down into various functions.

int main() {

  ////////// THE INITIALISATION iS HERE //////////

  /* Here, we set the initial distribution functions. Following the suggestion in Krüger Sec. 3.4.7.5 (Pgs. 92-93), I'm "unrolling" the loop over momentum space, and calculating f[i][j][k = 1] to
   * f[i][j][k = 9] explicitly. According to Krüger, this makes the calculations faster. These also seem to be more accurate for some reason. These equations are given by expanding Eq. 3.54 (Pg. 82)
   * in Krüger (Sec. 3.4.5). These can be further simplified to the equations given in the aforementioned section in Krüger (specifically, on Pg. 93, Eq. 3.65). */
  for (int i = 0; i < x_len; i++) {
    for (int j = 0; j < y_len; j++) {
      vel_sq[i][j] = std::pow(ux[i][j], 2) + std::pow(uy[i][j], 2);

      f_eq[i][j][0] = (4.0/9.0) * rho[i][j] * (1.0 - (vel_sq[i][j] / (2 * cs_sq)));
      f_eq[i][j][1] = (1.0/9.0) * rho[i][j] * (1.0 + (ux[i][j]) / cs_sq + (std::pow(ux[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
      f_eq[i][j][2] = (1.0/9.0) * rho[i][j] * (1.0 + (uy[i][j]) / cs_sq + (std::pow(uy[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
      f_eq[i][j][3] = (1.0/9.0) * rho[i][j] * (1.0 - (ux[i][j]) / cs_sq + (std::pow(ux[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
      f_eq[i][j][4] = (1.0/9.0) * rho[i][j] * (1.0 - (uy[i][j]) / cs_sq + (std::pow(uy[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
      f_eq[i][j][5] = (1.0/36.0) * rho[i][j] * (1.0 + (ux[i][j] + uy[i][j]) / cs_sq + (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
      f_eq[i][j][6] = (1.0/36.0) * rho[i][j] * (1.0 - (ux[i][j] - uy[i][j]) / cs_sq - (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
      f_eq[i][j][7] = (1.0/36.0) * rho[i][j] * (1.0 - (ux[i][j] + uy[i][j]) / cs_sq + (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
      f_eq[i][j][8] = (1.0/36.0) * rho[i][j] * (1.0 + (ux[i][j] - uy[i][j]) / cs_sq - (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
    }
  }
  f = f_eq;
  f_prop = f_eq;

  // In the streaming step, new_x and new_y are indices used to update the distribution; they're declared up-front here to avoid Visual Studio getting annoyed with me.
  int new_x;
  int new_y;


  ////////// THE MAIN SIMULATION LOOP STARTS HERE //////////

  for (int t = 0; t < t_steps; t++) {
    /* Here, we compute the macroscopic quantities (density and the velocity fields) at each time step. Again, we follow the suggestion in Krüger (Sec. 3.3.3.3 this time, Pg. 69) to unroll the
     * calculations in momentum space and perform each calculation explicitly. As before, I have a good handle on how the loop in momentum space would be input here; I'm just doing this at Krüger's
     * suggestion. These expressions are taken from the aforementioned section in Krüger (specifically, on Pg. 69, Eq. 3.12). */
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        rho[i][j] = f_prop[i][j][0] + f_prop[i][j][1] + f_prop[i][j][2] + f_prop[i][j][3] + f_prop[i][j][4] + f_prop[i][j][5] + f_prop[i][j][6] + f_prop[i][j][7] + f_prop[i][j][8];
        ux[i][j] = (f_prop[i][j][1] + f_prop[i][j][5] + f_prop[i][j][8] - f_prop[i][j][3] - f_prop[i][j][6] - f_prop[i][j][7]) / rho[i][j];
        uy[i][j] = (f_prop[i][j][2] + f_prop[i][j][5] + f_prop[i][j][6] - f_prop[i][j][4] - f_prop[i][j][7] - f_prop[i][j][8]) / rho[i][j];

//        sum_rho = 0.0;
//        for (int k = 0; k < q_num; k++) {
//          sum_rho += f_prop[i][j][k];
//          rho[i][j] = sum_rho;
//          ux[i][j] = (f_prop[i][j][1] + f_prop[i][j][5] + f_prop[i][j][8] - f_prop[i][j][3] - f_prop[i][j][6] - f_prop[i][j][7]) / rho[i][j];
//          uy[i][j] = (f_prop[i][j][2] + f_prop[i][j][5] + f_prop[i][j][6] - f_prop[i][j][4] - f_prop[i][j][7] - f_prop[i][j][8]) / rho[i][j];
//        }
      }
    }


    /* Here, we compute the equilibrium distribution at the given timestep. As before, this explicitly unrolls the loop over momentum space in the calculation of the distribution functions, by
     * expanding Eq. 3.54 (Pg. 82) in Krüger (Sec. 3.4.5). These can, as always, be further simplified to the equations given in Eq. 3.65 (Pg. 93) in Krüger. For completeness, we also show what the
     * expression in the k loop would look like.*/
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        vel_sq[i][j] = std::pow(ux[i][j], 2) + std::pow(uy[i][j], 2);

        f_eq[i][j][0] = (4.0/9.0) * rho[i][j] * (1.0 - (vel_sq[i][j] / (2 * cs_sq)));
        f_eq[i][j][1] = (1.0/9.0) * rho[i][j] * (1.0 + (ux[i][j]) / cs_sq + (std::pow(ux[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
        f_eq[i][j][2] = (1.0/9.0) * rho[i][j] * (1.0 + (uy[i][j]) / cs_sq + (std::pow(uy[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
        f_eq[i][j][3] = (1.0/9.0) * rho[i][j] * (1.0 - (ux[i][j]) / cs_sq + (std::pow(ux[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
        f_eq[i][j][4] = (1.0/9.0) * rho[i][j] * (1.0 - (uy[i][j]) / cs_sq + (std::pow(uy[i][j], 2)) / (2 * std::pow(cs_sq, 2)) - (vel_sq[i][j]) / (2 * cs_sq));
        f_eq[i][j][5] = (1.0/36.0) * rho[i][j] * (1.0 + (ux[i][j] + uy[i][j]) / cs_sq + (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
        f_eq[i][j][6] = (1.0/36.0) * rho[i][j] * (1.0 - (ux[i][j] - uy[i][j]) / cs_sq - (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
        f_eq[i][j][7] = (1.0/36.0) * rho[i][j] * (1.0 - (ux[i][j] + uy[i][j]) / cs_sq + (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);
        f_eq[i][j][8] = (1.0/36.0) * rho[i][j] * (1.0 + (ux[i][j] - uy[i][j]) / cs_sq - (ux[i][j] * uy[i][j]) / (std::pow(cs_sq, 2)) + (vel_sq[i][j]) / cs_sq);

//        for (int k = 0; k < q_num; k++) {
//          u_dot_ci[i][j][k] = (ux[i][j] * cx_float[k] + uy[i][j] * cy_float[k]);
//          f_eq[i][j][k] = w[k] * rho[i][j] * (1 + u_dot_ci[i][j][k] / cs_sq + std::pow(u_dot_ci[i][j][k], 2) / (2 * std::pow(cs_sq, 2)) - vel_sq[i][j] / (2 * cs_sq));
//        }

      }
    }


    // Here, we perform the collision step. THIS DOES NOT YET INCLUDE A VARIABLE TIMESTEP PARAMETER.
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        for (int k = 0; k < q_num; k++) {
          f[i][j][k] = (1 - (1.0/tau)) * f_prop[i][j][k] + (1.0/tau) * f_eq[i][j][k];
        }
      }
    }


    /* Here, we perform the streaming step, without applying the boundary condition at the wall, but with applying the periodic boundary conditions. The boundary condition at the wall shifts
     * everything over, and this will be applied momentarily. */
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        for (int k = 0; k < q_num; k++) {
          new_x = (i + cx_int[k] + x_len) % x_len;
          new_y = (j + cy_int[k] + y_len) % y_len;
          f_prop[new_x][new_y][k] = f[i][j][k];
        }
      }
    }


    /* Here, we apply the macroscopic boundary conditions. As mentioned in Krüger Sec. 5.3.4.1 (Pg. 190), since the lattice Boltzmann equation doesn't directly deal with the macroscopic fields (density
     * and velocity), we need to explicitly impose the continuity equation, the no-slip condition, and the no-penetration condition. The wall density is straightforwardly calculated in the
     * aforementioned section of Krüger. However, we note that there is an explicit de-linking of timestep and space-step in this derivation. Thus, for a speed c = dx/dt for timestep dt and space step
     * dx, expressions that should be c/(c ± u_wall) are instead written here as 1/(1 ± u_wall). This uses Eq. 5.32 and Eq. 5.33 (Pg. 191) in Krüger. */
     for (int i = 0; i < x_len; i++) {
       // Here, we apply the boundary conditions for the bottom wall: continuity, no-slip (ux[i][0] = u_wall_bottom), and no-penetration (uy[i][0] = 0.0).
       rho[i][0] = (1.0/(1.0 - uy[i][0])) * (f_prop[i][0][0] + f_prop[i][0][1] + f_prop[i][0][3] + 2.0 * (f_prop[i][0][4] + f_prop[i][0][7] + f_prop[i][0][8]));
       ux[i][0] = u_wall_bottom;
       uy[i][0] = 0.0;

       // Here, we apply the boundary conditions for the top wall: continuity, no-slip (ux[i][0] = u_wall_top), and no-penetration (uy[i][0] = 0.0).
       rho[i][y_len - 1] = (1.0/(1.0 + uy[i][y_len - 1])) * (f_prop[i][y_len - 1][0] + f_prop[i][y_len - 1][1] + f_prop[i][y_len - 1][3]
                                                          + 2.0 * (f_prop[i][y_len - 1][2] + f_prop[i][y_len - 1][5] + f_prop[i][y_len - 1][6]));
       ux[i][y_len - 1] = u_wall_top;
       uy[i][y_len - 1] = 0.0;
     }


    /* Here, we apply the boundary conditions onto the newly-streamed f_prop. Here, we use the non-equilibrium (second-order) expressions in the wet-node layout. The primary focus is on the Zou-He
     * (nonequilibrium bounce back) boundary conditions (c.f. Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48). For completeness, we also provide the boundary conditions used for the nonequilibrium
     * extrapolation method (c.f. Krüger Sec. 5.3.4.3, Pgs. 194-196, Eq. 5.40-5.41; derived from Eq. 4.35 taken from Sec. 4.2.4 in Krüger, Pgs. 118-119.) */
    for (int i = 0; i < x_len; i++) {

      /* Here, we incorporate the Zou-He (nonequilibrium bounce back) method, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48. This first applies the ZH boundary conditions on the
       * bottom plate. */
      f_prop[i][0][2] = f_prop[i][0][4] + (2.0 * rho[i][0] * uy[i][0])/3.0;
      f_prop[i][0][5] = f_prop[i][0][7] + (rho[i][0] * uy[i][0])/6.0 - (f_prop[i][0][1] - f_prop[i][0][3])/2.0 + ux[i][0]/2.0;
      f_prop[i][0][6] = f_prop[i][0][8] + (rho[i][0] * uy[i][0])/6.0 + (f_prop[i][0][1] - f_prop[i][0][3])/2.0 - ux[i][0]/2.0;

      /* Here, we incorporate the Zou-He (nonequilibrium bounce back) method, discussed in Krüger Sec. 5.3.4.4, Pgs. 196-199, Eq. 5.42-5.48. This now applies the ZH boundary conditions on the top
       * plate. */
      f_prop[i][y_len - 1][4] = f_prop[i][y_len - 1][2] - (2.0 * rho[i][y_len - 1] * uy[i][y_len - 1])/3.0;
      f_prop[i][y_len - 1][7] = f_prop[i][y_len - 1][5] - (rho[i][y_len - 1] * uy[i][y_len - 1])/6.0 + (f_prop[i][y_len - 1][1] - f_prop[i][y_len - 1][3])/2.0 - ux[i][y_len - 1]/2.0;
      f_prop[i][y_len - 1][8] = f_prop[i][y_len - 1][6] - (rho[i][y_len - 1] * uy[i][y_len - 1])/6.0 - (f_prop[i][y_len - 1][1] - f_prop[i][y_len - 1][3])/2.0 + ux[i][y_len - 1]/2.0;


//      /* Here, we incorporate the nonequilibrium extrapolation method, discussed in Krüger Sec. 5.3.4.3, Pgs. 194-196, Eq. 5.40-5.41. Those equations are originally derived from Sec. 4.2.4 in
//       * Krüger, Pgs. 118-119.) THIS DOES NOT YET INCORPORATE THE MULTIPLICATIVE FACTOR OF TAU. */
//      for (int k = 0; k < q_num; k++) {
//        u_dot_ci[i][0][k] = (ux[i][0] * cx_float[k] + uy[i][0] * cy_float[k]);
//        u_dot_ci[i][y_len - 1][k] = (ux[i][y_len - 1] * cx_float[k] + uy[i][y_len - 1] * cy_float[k]);
//        f_prop[i][0][k] = w[k] * (rho[i][0] + u_dot_ci[i][0][k]/cs_sq) + (f_prop[i][1][k] - f_eq[i][1][k]);
//        f_prop[i][y_len - 1][k] = w[k] * (rho[i][y_len - 1] + u_dot_ci[i][y_len - 1][k]/cs_sq) + (f_prop[i][y_len - 2][k] - f_eq[i][y_len - 2][k]);
//      }
    
    }


  }


  /* Here, we output the velocities and the densities, to test against the analytic solution. The problem has infinite symmetry in the x-direction, and the velocity depends *only* on the profile in
   * the y-direction. In particular, the analytic solution should be ux = ux(y) = ux_init * y / y_len (where y is a variable), uy = 0, and rho = 1 (since this is an incompressible problem). */
  std::cout << "ux(y)" << std::endl;
  for (int i = 0; i < x_len; i++) {
    std::cout << "[";
    for (int j = 0; j < y_len; j++) {
      std::cout << ux[i][j] << " ";
    }
    std::cout << "]" << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;


  // Output 
  std::cout << "uy" << std::endl;
  for (int i = 0; i < x_len; i++) {
    std::cout << "[";
    for (int j = 0; j < y_len; j++) {
      std::cout << uy[i][j] << " ";
    }
    std::cout << "]" << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;


  // Output densities
  std::cout << "rho" << std::endl;
  for (int i = 0; i < x_len; i++) {
    std::cout << "[";
    for (int j = 0; j < y_len; j++) {
      std::cout << rho[i][j] << " ";
    }
    std::cout << "]" << std::endl;
  }


}
