This folder contains some `run configurations` for the PyCharm ide.
To use them, you do the following steps:

- Open the mlonmcu folder as a project in PyCharm. 
  The run configuration will be automatically searched and loaded by PyCharm

- Select the right python interpreter (the path to a virtual environment or the path to a conda environment) for this project in PyCharm, create a new one if it does not exist.

- Install the mlonmcu project as an editable project using the following command in the terminal in the PyCharm ide.
    ```
    pip install -e .
    ```

- On the top right of the PyCharm ide. select the run configuration `step1: init env` and click the run button (green triangle). 
    
    > 📝
    > It is equivalent to the command 
    > ```
    >  python -m mlonmcu.cli.main init -t pulp -n pulp
    > ```
    > 
    > `-t pulp` means using the template `./resourses/templates/pulp.yml.j2` as the environment configuration
    > 
    > `-n pulp` means using `pulp` as the name of the environment

    A `run` window should pop up at the bottom of the PyCharm ide, And you will be asked for approval to continue.
    ```
    Initializing ML on MCU environment
    Selected target directory: /home/$USER/.config/mlonmcu/environments/pulp
    The directory does not exist! - Create directory? [Y/n] 
    ```
    approve and wait until the initialization finishes. 
    > 📝
    > if you want to find out how to change the target directory, 
    > You can use the command 
    > ```
    > python -m mlonmcu.cli.main init --help
    > ```
    > to find out how to do that
    
    You should remember the `target directory` and set it as the value of the environmental variable `MLONMCU_HOME` in all of the run configurations. The top answer of [this post](https://stackoverflow.com/questions/42708389/how-to-set-environment-variables-in-PyCharm) shows how to set the environment variables in the run configuration in PyCharm.

- Now you are ready to go. select the run configuration `step2: setup` and click the run button (green triangle) to populate the environment folder. It will take about half hour to finish.

- Finally, select the run configuration `step3: run resnet on pulpxpulpissimo` and click the run button (green triangle) to run an example.

    This example will run model resnet on pulp target with xpulp extension in the gvsoc simulator.

    > 📝
    > It is equivalent to the command 
    > ```
    > MLONMCU_HOME=<you environment location>; \
    > python -m mlonmcu.cli.main flow run resnet -t gvsoc_pulp
    > -c gvsoc_pulp.model=pulpissimo -f xpulp -c xpulp_version=2 \
    > -c mlif.print_outputs=1 -c gvsoc_pulp.print_outputs=1 -v
    > ```
    > 
    > `-t gvsoc_pulp` means using `gvsoc_pulp` as the target.
    > 
    > `-c gvsoc_pulp.model=pulpissimo` means using the pulpissimo model 
    > instead of pulp model for the target.
    > 
    > `-f xpulp` means using `xpulp` ISA extension.
    >
    > `-c xpulp_version=2` means using the version2 of `xpulp`
    > 
    > `-c mlif.print_outputs=1` means showing the mlif (Machine Learning Interface) informations in the 
    > terminal
    > 
    > `-v` means showing the commands executed by mlonmcu in the terminal

    

