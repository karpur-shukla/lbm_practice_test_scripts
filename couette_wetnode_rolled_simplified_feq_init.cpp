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
 *   2nd order (noneq) BCs: None
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
 * - NEED TO GO FROM WET-NODE BOUNCE-BACK BCS WITH NO 2ND ORDER TO HALFWAY BOUNCE-BACK BCS WITH 2ND ORDER. THIS REQUIRED UNDERSTANDING HOW TO SHIFT THE GRID SO THAT WE HAVE THE 1/2 OFFSET.
 *   -- UPDATE EVERYWHERE! :c
 * - NEED TO UPDATE THE EQUILIBRIUM DISTRIBUTION INITIALISATION AND UPDATE TERMS TO INCLUDE A VARIABLE SPEED OF SOUND.
 *   -- UPDATE AT LINES 132-150, 170-186,
 * - NEED TO HAVE A PROOF-OF-CONCEPT OF THE INPUT THE GENERAL EQUATIONS 3.4 AND 3.54 IN KRÜGER, RATHER THAN THE SIMPLIFIED VERSION (EVEN THOUGH THIS IS THE FASTER VERSION.)
 * - NEED TO IMPLEMENT ZOU-HE/NON-EQUILIBRIUM BOUNCE-BACK (c.f. Krüger Pgs. 194-199).
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
int x_len = 40;                   // number of x grid points. System is periodic in x.
int y_len = 20;                   // number of y grid points
int t_steps = 10001;           // number of time steps. Since I update the densities and velocities at the beginning of the simulation, I'm doing one extra timestep.
double tau = 0.9;                 // BGK relaxation time
double omega = 1.0/tau;           // BGK inverse relaxation time (lowercase omega, NOT capital omega)
double rho_init = 1.0;            // initial density
double cs = 1/3.0;              // speed of sound in lattice units
double cs_sq = std::pow(cs, 2);   // speed of sound squared
double cs_four = std::pow(cs, 4); // speed of sound to the fourth
double u_wall_top = 0.1;          // top wall velocity
double u_wall_bottom = 0.0;       // bottom wall velocity
double ux_init = 0.0;             // global initial velocity in the x-direction
double uy_init = 0.0;             // global initial velocity in the y-direction

double vel_init_sq = std::pow(ux_init, 2) + std::pow(uy_init, 2); // square of the magnitude of the initial velocity

/* These are the lattice parameters. We define the x-direction and y-direction lattice speeds as both a float vector and an int vector. The float vector can go into an equation for the */
int q_num = 9; // number of velocity directions (Q in DnQm). Here, Q = 9, with q = 0 as self-velocity.

std::vector<float> w = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; // lattice weights
std::vector<int> cx_int = {0, 1, 0, -1,  0, 1, -1, -1,  1};                       // x-direction lattice speeds
std::vector<int> cy_int = {0, 0, 1,  0, -1, 1,  1, -1, -1};                       // y-direction lattice speeds
//std::vector<float> cx_float = {0.0, 1.0, 0.0, -1.0,  0.0, 1.0, -1.0, -1.0,  1.0}; // x-direction lattice speeds
//std::vector<float> cy_float = {0.0, 0.0, 1.0,  0.0, -1.0, 1.0,  1.0, -1.0, -1.0}; // y-direction lattice speeds

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


  ///* This creates the vector of nodes, which are offset from the "regular" grid by 1/2 in each direction. These will be used later, to implement the
  //   half-way bounce-back conditions, when I understand how that works. */
  //vect1D_float x_node(x_len);
  //for (int i = 0; i < x_len + 1; i++) {
  //  x_node[i] = i + 0.5;
  //}
  //vect1D_float y_node(y_len);
  //for (int j = 0; j < y_len + 1; j++) {
  //  y_node[j] = j + 0.5;
  //}


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


////////// THE MAIN SIMULATION LOOP STARTS HERE //////////

// Right now, the entire LB simulation (initialisation included) is done in main(). This will eventually be broken down into various functions.

int main() {


  ////////// THE INITIALISATION iS HERE //////////

  /* Here, we set the initial distribution functions. Following the suggestion in Krüger Sec. 3.4.7.5 (Pgs. 92-93), I'm "unrolling" the loop over momentum space, and calculating f[i][j][k = 1] to
   * f[i][j][k = 9] explicitly. According to Krüger, this makes the calculations faster. These expressions are taken from the aforementioned section in Krüger (specifically, on Pg. 93, Eq. 3.65).
   * THIS DOES NOT YET INCORPORATE A VARIABLE SPEED OF SOUND. */
  for (int i = 0; i < x_len; i++) {
    for (int j = 0; j < y_len; j++) {
      for (int k = 0; k < q_num; k++) {
        f_eq[i][j][k] = w[k]
      }
//      f_eq[i][j][0] = ((2 * rho_init)/9.0) * (2.0 - 3.0 * vel_init_sq);
//      f_eq[i][j][1] = (rho_init/18.0) * (2.0 + 6.0 * ux_init + 9.0 * std::pow(ux_init, 2) - 3.0 * vel_init_sq);
//      f_eq[i][j][2] = (rho_init/18.0) * (2.0 + 6.0 * uy_init + 9.0 * std::pow(uy_init, 2) - 3.0 * vel_init_sq);
//      f_eq[i][j][3] = (rho_init/18.0) * (2.0 - 6.0 * ux_init + 9.0 * std::pow(ux_init, 2) - 3.0 * vel_init_sq);
//      f_eq[i][j][4] = (rho_init/18.0) * (2.0 - 6.0 * uy_init + 9.0 * std::pow(uy_init, 2) - 3.0 * vel_init_sq);
//      f_eq[i][j][5] = (rho_init/36.0) * (1.0 + 3.0 * (ux_init + uy_init) + 9.0 * (ux_init * uy_init) + 3.0 * vel_init_sq);
//      f_eq[i][j][6] = (rho_init/36.0) * (1.0 - 3.0 * (ux_init - uy_init) - 9.0 * (ux_init * uy_init) + 3.0 * vel_init_sq);
//      f_eq[i][j][7] = (rho_init/36.0) * (1.0 - 3.0 * (ux_init + uy_init) + 9.0 * (ux_init * uy_init) + 3.0 * vel_init_sq);
//      f_eq[i][j][8] = (rho_init/36.0) * (1.0 + 3.0 * (ux_init - uy_init) - 9.0 * (ux_init * uy_init) + 3.0 * vel_init_sq);
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
      }
    }


    /* Here, we compute the equilibrium distribution at the given timestep. As before, this explicitly unrolls the loop over momentum space in the
     * calculation of the distribution functions, using Eq. 3.65 (Pg. 93) in Krüger. THIS DOES NOT YET INCORPORATE A VARIABLE SPEED OF SOUND. */
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        vel_sq[i][j] = std::pow(ux[i][j], 2) + std::pow(uy[i][j], 2);
        for (int k = 0; k < q_num; k++) {
          f_eq[i][j][k] = w[k] * (rho[i][j] + 3 * (ux[i][j] * cx_float[k] + uy[i][j] * cy_float[k]));
        }
        
//        f_eq[i][j][0] = ((2 * rho[i][j])/9.0) * (2.0 - 3.0 * vel_sq[i][j]);
//        f_eq[i][j][1] = (rho[i][j]/18.0) * (2.0 + 6.0 * ux[i][j] + 9.0 * std::pow(ux[i][j], 2) - 3.0 * vel_sq[i][j]);
//        f_eq[i][j][2] = (rho[i][j]/18.0) * (2.0 + 6.0 * uy[i][j] + 9.0 * std::pow(uy[i][j], 2) - 3.0 * vel_sq[i][j]);
//        f_eq[i][j][3] = (rho[i][j]/18.0) * (2.0 - 6.0 * ux[i][j] + 9.0 * std::pow(ux[i][j], 2) - 3.0 * vel_sq[i][j]);
//        f_eq[i][j][4] = (rho[i][j]/18.0) * (2.0 - 6.0 * uy[i][j] + 9.0 * std::pow(uy[i][j], 2) - 3.0 * vel_sq[i][j]);
//        f_eq[i][j][5] = (rho[i][j]/36.0) * (1.0 + 3.0 * (ux[i][j] + uy[i][j]) + 9.0 * (ux[i][j] * uy[i][j]) + 3.0 * vel_sq[i][j]);
//        f_eq[i][j][6] = (rho[i][j]/36.0) * (1.0 - 3.0 * (ux[i][j] - uy[i][j]) - 9.0 * (ux[i][j] * uy[i][j]) + 3.0 * vel_sq[i][j]);
//        f_eq[i][j][7] = (rho[i][j]/36.0) * (1.0 - 3.0 * (ux[i][j] + uy[i][j]) + 9.0 * (ux[i][j] * uy[i][j]) + 3.0 * vel_sq[i][j]);
//        f_eq[i][j][8] = (rho[i][j]/36.0) * (1.0 + 3.0 * (ux[i][j] - uy[i][j]) - 9.0 * (ux[i][j] * uy[i][j]) + 3.0 * vel_sq[i][j]);
      }
    }


    // Here, we perform the collision step. THIS DOES NOT YET INCLUDE A VARIABLE TIMESTEP PARAMETER.
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        for (int k = 0; k < q_num; k++) {
          f[i][j][k] = (1 - (1/tau)) * f_prop[i][j][k] + (1/tau) * f_eq[i][j][k];
        }
      }
    }


    /* Here, we perform the streaming step, without applying the boundary condition at the wall, but with applying the periodic boundary conditions.
     * The boundary condition at the wall shifts everything over, and this will be applied momentarily. */
    for (int i = 0; i < x_len; i++) {
      for (int j = 0; j < y_len; j++) {
        for (int k = 0; k < q_num; k++) {
          new_x = (i + cx_int[k] + x_len) % x_len;
          new_y = (j + cy_int[k] + y_len) % y_len;
          f_prop[new_x][j][k] = f[i][j][k];
        }
      }
    }


    ///* Here, we apply the boundary conditions. As mentioned in Krüger Sec. 5.3.4.1 (Pg. 190), since the lattice Boltzmann equation doesn't directly deal with the macroscopic fields (density and
    // * velocity), we need to explicitly impose the continuity equation and the no-penetration condition. The wall density is straightforwardly calculated in the aforementioned section of Krüger.
    // * However, we note that there is an explicit de-linking of timestep and space-step in this derivation. Thus, for a speed c = dx/dt for timestep dt and space step dx, expressions that should
    // * be c/(c ± u_wall) are instead written here as 1/(1 ± u_wall). This uses Eq. 5.32 and Eq. 5.33 (Pg. 191) in Krüger. */
    //for (int i; i < x_len; i++) {
    //    ux[i][0] = u_wall_bottom;
    //    ux[i][y_len-1] = u_wall_top;
    //    rho[i][0] = 1.0 / (1.0 - u_wall_bottom) * (f_prop[i][0][0] + f_prop[i][0][1] + f_prop[i][0][3] + 2 * (f_prop[i][0][4] + f_prop[i][0][7] + f_prop[i][0][8]));
    //    rho[i][y_len] = 1.0 / (1.0 + u_wall_top) * (f_prop[i][y_len][0] + f_prop[i][y_len][1] + f_prop[i][y_len][3] + 2 * (f_prop[i][y_len][2] + f_prop[i][y_len][5] + f_prop[i][0][6]));
    //}

    /* Here, we apply the boundary conditions onto the newly-streamed f_prop. This uses the bounce back scheme, modified for the wet-node layout (as was done here). Here, it's easier to use the
     * explicit expressions given in Eq. 5.25, 5.27, and 5.28 in Krüger, rather than the general expression Eq. 5.26 in Krüger. Like with the "unrolled" expressions for rho, u_x, and u_y; this is not
     * done because implementing those expressions is particularly challenging.*/
    for (int i = 0; i < x_len; i++) {
      // Bounce-back on the bottom plate
      f_prop[i][0][2] = f_prop[i][0][4];
      f_prop[i][0][5] = f_prop[i][0][7];
      f_prop[i][0][6] = f_prop[i][0][8];
    
      // Bounce-back on the top plate, with the wall motion
      f_prop[i][y_len - 1][4] = f_prop[i][y_len - 1][2];
      f_prop[i][y_len - 1][7] = f_prop[i][y_len - 1][5] - u_wall_top/6.0;
      f_prop[i][y_len - 1][8] = f_prop[i][y_len - 1][6] + u_wall_top/6.0;
    }


  }

  // Here, we output the x-velocities, to test against the analytic solution. We can pick any of the 
  std::cout << "ux(y)";
  for (int j = 0; j < y_len; j++) {
    std::cout << ux[10][j] << std::endl;
  }
}


//// Boundary condition (wet node)
//// Setting macroscopic quantities at boundaries
//// Bottom wall (rest)
//rho[0][0] = 1 / (1 - v[0][0]) * (fprop[0][0][8] + fprop[0][0][0] + fprop[0][0][2] + 2 * (fprop[0][0][3] + fprop[0][0][6] + fprop[0][0][7]));
//u[0][0] = 0;
//v[0][0] = 0;
//
//// Top wall (moving)
//rho[0][NY - 1] = 1 / (1 + v[0][NY - 1]) * (fprop[0][NY - 1][8] + fprop[0][NY - 1][0] + fprop[0][NY - 1][2] + 2 * (fprop[0][NY - 1][1] + fprop[0][NY - 1][5] + fprop[0][NY - 1][6]));
//u[0][NY - 1] = u_max;
//v[0][NY - 1] = 0;
//
//// Setting populations quantities at boundaries
//if (wetnode == 1) {         // 1) equilibrium scheme BC
//    for (int k = 0; k < NPOP; ++k) {
//        fprop[0][0][k] = w[k] * (rho[0][0] + 3 * (cx[k] * u[0][0] + cy[k] * v[0][0]));
//        fprop[0][NY - 1][k] = w[k] * (rho[0][NY - 1] + 3 * (cx[k] * u[0][NY - 1] + cy[k] * v[0][NY - 1]));
//    }
//}
//else if (wetnode == 2) {  // 2) non-equilibrium extrapolation method BC
//    for (int k = 0; k < NPOP; ++k) {
//        fprop[0][0][k] = w[k] * (rho[0][0] + 3 * (cx[k] * u[0][0] + cy[k] * v[0][0])) + (fprop[1][0][k] - feq[1][0][k]);
//        fprop[0][NY - 1][k] = w[k] * (rho[0][NY - 1] + 3 * (cx[k] * u[0][NY - 1] + cy[k] * v[0][NY - 1])) + (fprop[NX - 2][NY - 1][k] - feq[NX - 2][NY - 1][k]);
//    }
//}
//else {                    // 3) non-equilibrium bounce-back method BC (note: rho=1)
//    fprop[0][0][2] = fprop[0][0][4] + 2.0 / 3.0 * v[0][0];
//    fprop[0][0][5] = fprop[0][0][7] + 1.0 / 6.0 * v[0][0] - 0.5 * (fprop[0][0][0] - fprop[0][0][2]) + 0.5 * u[0][0];
//    fprop[0][0][6] = fprop[0][0][8] + 1.0 / 6.0 * v[0][0] + 0.5 * (fprop[0][0][0] - fprop[0][0][2]) - 0.5 * u[0][0];
//
//    fprop[0][NY - 1][4] = fprop[0][NY - 1][2] - 2.0 / 3.0 * v[0][NY - 1];
//    fprop[0][NY - 1][7] = fprop[0][NY - 1][5] - 1.0 / 6.0 * v[0][NY - 1] + 0.5 * (fprop[0][NY - 1][0] - fprop[0][NY - 1][2]) - 0.5 * u[0][NY - 1];
//    fprop[0][NY - 1][8] = fprop[0][NY - 1][6] - 1.0 / 6.0 * v[0][NY - 1] + 0.5 * (fprop[0][NY - 1][2] - fprop[0][NY - 1][0]) + 0.5 * u[0][NY - 1];
//}
