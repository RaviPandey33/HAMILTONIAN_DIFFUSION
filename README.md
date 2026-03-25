## The idea was generated while reading a paper by quest lab, titled "Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting" and from my own work during my masters.

In my work titled - "Optimizing ODE Solvers: A Machine Learning Approach to RK4 Enhancement", i worked with Hamiltonian equations, by building a loss function which takes into account the energy error of the Hamiltonian. Later i compared the optimized integrator on various parameters, inlcuding a comparision study, between the loss function with no energy term vs loss function with energy term.

I used this same idea to try to generate trajectories using diffusion models, which follows the energy conservation property of the hamiltonian, while maintaing the dynamics.
I started with simple diffusion error, later added energy term in the loss function and then later added the dynamics error as well.

The generated trajectories can be checked in the "experiments/" folder, for different settings, whose details are also mentioned. 

Sadly the idea failed to generate perfect circles(which signifies that energy has been conserved). Future way ahead could be to try changing the loss function in some other ways where it retains the systems phycial properties.



To run the project, clone the repo in your local computer, and run the command :
"python run_pipeline.py"

Changes to the settings can be made through the config.py file and also through the run_pipeline.py file.
