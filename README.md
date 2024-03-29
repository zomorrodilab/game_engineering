# Setup

1. **Install Gurobi**: You will need to install the Gurobi solver software and obtain a license. The Gurobi solver is freely available for academic users that are affiliated with academic institutions. You can find instructions on how to do this [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
   
   If you cannot obtain an academic license, you can use another solver by changing the line `import optlang.gurobi_interface` to `import optlang.cplex_interface` or `import optlang.glpk_interface` in the `manuscript_code.ipynb` file. The GLPK solver is freely available, while the CPLEX solver is free for academic users that are affiliated with academic institutions.

2. **Install Conda and Jupyter Lab**: We recommend that you follow the instructions below, which will guide you through the process of setting up a Conda environment and running the code using Jupyter Lab. This will ensure that the results are accurately reproducible. However, you can also install the required packages listed in `environment.yml` manually using pip and run the code using Jupyter Notebook or any other IDE that supports Jupyter notebooks, but this does not guarantee that the results will be accurately reproducible since the package versions may differ.

    If you don't have Conda installed, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html). We recommend using the Miniconda distribution. Then, you can then install Jupyter Lab by running the following command in your terminal:

    ```bash
    conda install -c conda-forge jupyterlab
    ```

3. **Create a Conda environment**: You can do this by running the following command in your terminal:

    ```bash
    conda env create -f environment.yml
    ipython kernel install --user --name=manuscript
    ```
    Now we have created a Conda environment called `manuscript` and a Jupyter kernel with the same name.

# Running the code
You can now run the Jupyter notebook by running the following command in your terminal:
```bash
jupyter notebook
```
Now you can navigate to either the `2-player games` or `4-player games` folder. Once you are in the desired folder, open the `.ipynb` file. 
For the last step, make sure to select the `manuscript` kernel that you created in the previous step by clicking on `Kernel` -> `Change kernel` -> `manuscript`.
Note: If you usually use IDEs like VSCode, you can also use the Jupyter extension to run the notebook directly from there.

Now that you are in the Jupyter notebook and have the correct environment running, you can run the code by clicking on `Cell` -> `Run All`. This will run all the cells in the notebook and produce the results.