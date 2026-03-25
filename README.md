
# This project has been created out of my curiosity of learn about the Diffusion mechanism during image generation. 

## The idea was generated while reading a paper by quest lab, titled "Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting" and from my own thesis during my masters which i recently finished. 

In my thesis report, titled - "Optimizing ODE Solvers: A Machine Learning Approach to RK4 Enhancement", i worked with Hamiltonian equations, by building a loss function which takes into account the energy error of the Hamiltonian. Later i compared the optimized integrator on various parameters, inlcuding a comparision study, between the loss function with no energy term vs loss function with energy term.

I used this same idea to try to generate trajectories using diffusion models, which follows the energy conservation property of the hamiltonian, while maintaing the dynamics.
I started with simple diffusion error, later added energy term in the loss function and then later added the dynamics error as well.

The generated trajectories can be checked in the "plots/" folder, for different settings, whose details are also mentioned. 

Sadly the idea failed to generate perfect circles(which signifies that energy has been conserved). Future way ahead could be to try changing the loss function in some other ways. For example, i can try to train the model in a way that the generated points are within a circle, which is what we want( along with the dynamics). 

Moreover one simple method for further testing could be to work with the 3 different parameters in present loss function. 
Right now i have given equal preferenes to all the error terms, namely, the diffusion error, the energy error, and the dynamics error. Maybe i can try later to give more priority to the energy error and the dynamics error, so that the generated points follow a circular curvature while maintaining the proper dynamics.

At the end, though i am not very sure if the above mentioned approach is a noval approach and whether or not this will work, but a more through experimentation could be justified.