# End-to-End ML OPS project for Weather prediction
Repo for the MSDS MLOps course

The purpose of this project is feed a simple Prohpet model the last twenty year on weather data in the city of Alicante (Spain), train it in a Metaflow flow and choose the best model parameters using MLflow. The best model for each weather variable will be accessible to a FastAPI app so users can ask what will be the weather prediction for a variable in the future.

### The Architecture 

This basic graph summarizes the framework and tools used in this project:

![My framework](./img/framework.png)



