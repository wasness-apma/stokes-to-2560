/// Poisson equation in MFEM
/// with pure Neumann boundary conditions

#include "mfem.hpp"
#include <cmath>

using namespace mfem;
using namespace std;

const real_t pi = M_PI;

const real_t alpha_under = 1e-6;
const real_t alpha_over = 1e6;
const real_t q = 0.1;

real_t mu = 1.0;
void u_exact_straightflow(const Vector &x, Vector & u);
void u_exact_curvilinear(const Vector &x, Vector & u);

void f_exact(const Vector &x, Vector &f);

real_t density_straightflow(const Vector &x);
real_t density_curvilinear(const Vector &x);

real_t alpha(const real_t rho);

SparseMatrix ToRowMatrix(LinearForm &lf);

int main(int argc, char *argv[])
{
   int order = 1;
   int ref_levels = 1;
   bool vis = false;
   bool curvilinear = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Polynomial order for the finite element space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform refinements.");
   args.AddOption(&vis, "-v", "--visualize", "--no-vis", "--no-visualization",
                  "-v 1 to visualize the solution.");
   args.AddOption(&curvilinear, "-c", "--curvilinear", "--curvilinear", "--curvilinear",
                  "-c 1 to do curvilinear solution");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   MFEM_ASSERT(order >= 1, "Order must be at least 1.");

   // Initialize MFEM
   Mesh mesh = mfem::Mesh::MakeCartesian2D(8, 8, Element::Type::TRIANGLE, true);
   int dim = mesh.Dimension();

   L2_FECollection l2_coll(0, dim); 
   H1_FECollection h1_coll(order, dim);
   H1_FECollection h1_coll_op1(order + 1, dim); 

   FiniteElementSpace velocity_space(&mesh, &h1_coll_op1, dim=dim); // velocity space
   FiniteElementSpace velocity_component_space(&mesh, &h1_coll_op1); // velocity space. Appears as derivative so need more regularity.
   FiniteElementSpace pressure_space(&mesh, &h1_coll); // pressure space
   FiniteElementSpace density_space(&mesh, &l2_coll);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   for (int lv = 0; lv < ref_levels; lv++)
   {
      mesh.UniformRefinement();

      velocity_space.Update();
      velocity_component_space.Update();
      pressure_space.Update();
      density_space.Update();

      const int dof_velocity = velocity_space.GetVSize();
      const int dof_pressure = pressure_space.GetVSize();
      const int dof_density = density_space.GetVSize();

      // Apply boundary conditions on all external boundaries:
      Array<int> ess_tdof_list;
      velocity_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);


      // Define Block offsets.
      // The first block is the momentum equation, the second block is the continuity equation, and the third block is average zero condition
      Array<int> block_offsets(4);
      block_offsets[0] = 0;
      block_offsets[1] = dof_velocity;
      block_offsets[2] = dof_pressure;
      block_offsets[3] = 1;
      block_offsets.PartialSum();

      // true solution vector. The mt part is a copy past.
      BlockVector x(block_offsets), rhs(block_offsets);
      x = 0.0;
      rhs = 0.0;

      std::cout << "***********************************************************\n";
      std::cout << "RT Space dofs = " << dof_velocity << "\n";
      std::cout << "H1 Space dofs = " << dof_pressure << "\n";
      std::cout << "Density Space dofs = " << dof_density << "\n";
      std::cout << "dim(Diffusion) = " << block_offsets[1] - block_offsets[0] << "\n";
      std::cout << "dim(Div) = " << block_offsets[2] - block_offsets[1] << "\n";
      std::cout << "dim(Zero) = " << block_offsets[3] - block_offsets[2] << "\n";
      std::cout << "***********************************************************\n\n";

      // separate this for the sake of later
      GridFunction rho(&density_space);
      rho = 0.0;
      if (curvilinear) {
         FunctionCoefficient densityCoefficient(&density_curvilinear);
         rho.ProjectCoefficient(densityCoefficient); 
      } else {
         FunctionCoefficient densityCoefficient(&density_straightflow);
         rho.ProjectCoefficient(densityCoefficient); 
      }

      GridFunction u(&velocity_space, x.GetBlock(0));

      if (curvilinear) {
         VectorFunctionCoefficient bdrConditions(dim=dim, &u_exact_curvilinear);
         u.ProjectBdrCoefficient(bdrConditions, ess_bdr);
      } else {
         VectorFunctionCoefficient bdrConditions(dim=dim, &u_exact_straightflow);
         u.ProjectBdrCoefficient(bdrConditions, ess_bdr);
      }

      GridFunction ux(&velocity_component_space, x.GetBlock(0));
      GridFunction uy(&velocity_component_space, x.GetBlock(0), velocity_component_space.GetVSize());
      
      GridFunction p(&pressure_space, x.GetBlock(1));

      // define the linear form, witout boundary conditions
      LinearForm load(&velocity_space, rhs.GetBlock(0).GetData());
      VectorFunctionCoefficient load_cf(dim, f_exact);
      load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
      load.Assemble();

      // Define the bilinear form for mass integration and diffusion integration
      BilinearForm massDiffusionOperator(&velocity_space);

      ConstantCoefficient mu_coeff(mu);
      massDiffusionOperator.AddDomainIntegrator(new VectorDiffusionIntegrator(mu_coeff));

      GridFunctionCoefficient rho_coeff(&rho);
      TransformedCoefficient alpha_coeff(&rho_coeff, alpha);
      massDiffusionOperator.AddDomainIntegrator(new VectorMassIntegrator(alpha_coeff));

      massDiffusionOperator.Assemble();
      massDiffusionOperator.EliminateEssentialBC(ess_bdr, u, rhs.GetBlock(0));
      massDiffusionOperator.Finalize();

      MixedBilinearForm divergenceOperator(&velocity_space, &pressure_space);
      divergenceOperator.AddDomainIntegrator(new VectorDivergenceIntegrator);
      divergenceOperator.Assemble();
      divergenceOperator.EliminateTrialEssentialBC(ess_bdr, u, rhs.GetBlock(1));
      divergenceOperator.Finalize();

      SparseMatrix * transposeDivergenceOperator = Transpose(divergenceOperator.SpMat());

      LinearForm avg_zero(&pressure_space);
      ConstantCoefficient one_cf(1.0);
      avg_zero.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
      avg_zero.Assemble();

      SparseMatrix linearFormZero = ToRowMatrix(avg_zero);
      SparseMatrix* linearFormZeroTranspose; 
      linearFormZeroTranspose = Transpose(linearFormZero);

      BlockMatrix stokesOperator(block_offsets);
      stokesOperator.SetBlock(0,0, &massDiffusionOperator.SpMat());
      stokesOperator.SetBlock(0,1, transposeDivergenceOperator);
      stokesOperator.SetBlock(1,0, &divergenceOperator.SpMat());
      stokesOperator.SetBlock(1,2, linearFormZeroTranspose);
      stokesOperator.SetBlock(2,1, &linearFormZero);

      SparseMatrix * stokesMatrix = stokesOperator.CreateMonolithic();

      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*stokesMatrix);
      umf_solver.Mult(rhs, x);

      // 12. Create the grid functions u and p. Compute the L2 error norms.

      VectorFunctionCoefficient* ucoeff;
      if (curvilinear) {
         ucoeff = new VectorFunctionCoefficient(dim, u_exact_curvilinear);
      } else {
         ucoeff = new VectorFunctionCoefficient(dim, u_exact_straightflow);    
      }

      real_t err = u.ComputeL2Error(*ucoeff);
      avg_zero.SetSize(dof_pressure);
      avg_zero.Assemble();
      real_t mass_err = avg_zero(p);

      // Print the solution
      std::cout << "Error : " << err << std::endl;
      std::cout << "Avg   : " << mass_err << std::endl;

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout_u;
      sout_u.open(vishost, visport);
      sout_u.precision(8);
      sout_u << "solution\n" << mesh << u
            << "window_title 'Design physics u'" << flush;

      delete stokesMatrix;
      delete linearFormZeroTranspose;
      delete transposeDivergenceOperator;
      delete ucoeff;
   }
   return 0;
}

void print_array(Array<int> arr) {
   int len = sizeof(arr) / sizeof(*arr);
   for (int i = 0; i < len; i++) {
      if (i < len - 1)
         std::cout << arr[i] << " ";
      else
         std::cout << arr[i];
   }
}

void print_array(Array<float> arr) {
   int len = sizeof(arr) / sizeof(*arr);
   for (int i = 0; i < len; i++) {
      if (i < len - 1)
         std::cout << arr[i] << " ";
      else
         std::cout << arr[i];
   }
}


const real_t minbound = 0.3;
const real_t maxbound = 0.7;
const real_t normalizer = pow(2 / (maxbound - minbound), 2);

real_t inverse_parabola(const real_t y) {
   return std::max(0.0, -normalizer * (y - minbound) * (y - maxbound));
}

// horizontal flow, constant
void u_exact_straightflow(const Vector &x, Vector &u)
{
   // constant => divergence free
   u(0) = inverse_parabola(x(1));
   u(1) = 0;
}

// horizontal flow, constant
void u_exact_curvilinear(const Vector &x, Vector &u)
{
   // constant => divergence free
   real_t r = std::sqrt(x(0)*x(0) + x(1)*x(1));
   u(0) = inverse_parabola(r) * x(1);
   u(1) = inverse_parabola(r) * -x(0);
}

void f_exact(const Vector &x, Vector &f)
{
   f(0) = 0;
   f(1) = 0;
}

real_t density_straightflow(const Vector &x)
{
   if (minbound < x(1) && x(1) < maxbound) {
      return 1;
   } else {
      return 0;
   }
}

real_t density_curvilinear(const Vector &x)
{
   real_t r = std::sqrt(x(0)*x(0) + x(1)*x(1));
   if (inverse_parabola(r) > 0) {
      return 1; 
   } else {
      return 0;
   }
}

real_t alpha(const real_t rho) {
   if (rho < 0) {
      std::string errstring = "Rho value should not be negative: ";
      errstring += std::to_string(rho);
      throw std::runtime_error(errstring);
   }
   return alpha_over + (alpha_under - alpha_over) * rho * (1 + q) / (rho + q);
}

SparseMatrix ToRowMatrix(LinearForm &lf)
{
   const int size = lf.FESpace()->GetTrueVSize();
   Array<int> row_ptr({0,size});
   Array<int> col_ind(size);
   std::iota(col_ind.begin(), col_ind.end(), int());

   lf.Assemble();
   double * data = lf.GetData();
   int *i, *j;
   row_ptr.StealData(&i);
   col_ind.StealData(&j);
   return SparseMatrix(i, j, data, 1, size, true, false, true);
}
