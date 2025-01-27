---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

61

### Question 2
> **Enter the study number for each member in the group**
>
> Answer:

214997, s215002, s216169

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>s
> Answer:

In this project, we made use of the [Transformers](https://github.com/huggingface/transformers) library developed by Huggingface. This library includes the [t5-small model](https://huggingface.co/t5-small), a natural language processing (NLP) model designed for tasks like translating text. For this project, we employed the Trainer class from the PyTorch Lightning framework to finetune and evaluate the t5-small model using a subset of the English/German (en-de) [WMT19 dataset](https://huggingface.co/datasets/wmt19), which originates from the Fourth Conference on Machine Translation. Additionally, we used Weights and Biases (`wandb`) to manage the configuration file containing the model's hyperparameters and to log the training and validation losses. Utilizing the obtained model and its parameters given for the best run, we were able to translate an input of English text to German.


## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

Initially, we established a new conda environment, called 'dtu_mlops_group_61', running Python 3.12. This would ensure that all developers would be running on the same environment-type.

In the project, we used a requirements.txt as well as the pyproject.toml file to keep track of the necessary Python packages and environments to run the project code. Some Python packages in requirements.txt require a specific version to be downloaded, as to prevent certain incompatibilities. The requirements.txt was created using pipreqs and was manually updated regularly as the project progressed.


To get a complete copy of our development enviroment, one would have to run the following commands (assuming they have git installed):
```
git clone https://github.com/MagnusBeng2/dtu_mlops_group_61.git
cd ./dtu_mlops_exam_project
conda create -n dtu_mlops_group_61 python=3.11
pip install -r requirements.txt
dvc pull
pip install -e .
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

The initial repository was very similiar to the MLops cookiecutter template and most template folders were included. Only small adjustements were made along the way. Throughout the report lifecycle we tried adhering to the structure, however, as features were added, the structure became more blurred. Some examples are that the src folder was not used as intended, the models folder remained empty, the folder lightning_logs was added, as well as multiple config files to the root. Features were usually not configered to work with the template, which resulted in many logs and config files to be stored in the root. All in all the structure needs to be reworked.


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

Yes, for this project we established rules for quality and formatting, which was enforded by the Ruff linter in our continuous integration. Together with having a pre-commit continuous integration, this ensured compliance with the PEP8 standards. The pre-commit would specifically work by fixing trailing, whitespaces, missing newlines and improperly sorted imports before commits was conducted to the Git repository.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

We implemented 11 tests that were run successfully:

<ul>
  <li>def test_api_request_valid(mock_model):</li>
  <li>def test_dataset_format():</li>
  <li>def test_model_is_torch():</li>
  <li>def test_model_output():</li>
  <li>def test_steps():</li>
  <li>def test_training_loop():</li>
  <li>def test_train_model_arguments():</li>
  <li>def test_train_model_steps(mock_trainer, mock_model, mock_load_from_disk, mock_wandb):</li>
  <li>def test_set_seed():</li>
  <li>def test_get_next_version():</li>
  <li>def test_training_loop(mock_trainer, mock_load_from_disk, mock_wandb):</li>
</ul>

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/__init__.py                   0      0   100%
src/models/__init__.py            2      0   100%
src/models/model.py              70      2    97%   44, 46
src/models/predict_model.py      55     30    45%   22-31, 38-45, 52-55, 68-69, 79-94, 98
src/models/train_model.py        86     16    81%   35-36, 49, 55-56, 154-164, 167
-----------------------------------------------------------
TOTAL                           213     48    77%

The code coverage is just below 100% for the model.py script. Lines 44 and 46 are related to edge-case checks in the _init_.py. These checks handle invalid configurations (e.g., missing or invalid parameters) that are not triggered in the current test setup, as we assumed valid configurations were always to be provided.

The code coverage is 45% for predict_model.py. Test focus on testing the core logic of the model. Following elements of the script are not tested:

1. The functionality for loading a checkpoint is not tested.
2. Parts of the code that utilize external data files is not tested.
3. The if _name_ == '_main_' is executed outside the pytest context.
4. Command-line arguments aren't tested.
5. Error handling, logging or sanity checks or not tested.

The code coverage for train_model is 81%. Error handling and logging mechanisms were not tested on. Instead, the focused lay testing the core functionality of the argument parser, data loading and repoducibility.

### Question 9

> **Did your workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had a branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We utilized two major branches in this project, main, which we used as the developer branch, and the safe branch, which we updated with working versions of the repository. This ensured that we kept the number of branches to a minimum, while still having flexibility and allowing for better error-handling.

For this project, we did not implement branch protection rules. In retrospect, it would have been most advantageous to establish branch protection rules inside the GitHub repository, making sure that push and merge requests did not override the wrong files. However, we believed it would add additional complexity to project and since we were only 4 group members, we decided not using them. All group members made the pledge to ensure to pull new updates before pushing, ensuring an up-to-date local repository and no merge conflicts.

That being said, it is definitely the best practice to construct branch protection rules in one's repository, especially when many developers work on it. For future projects, it would be best to include branch protection rules early on.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

DVC was implemented, using a Google Cloud Service (GCS) bucket storage. As our data was static, the primary reason for using DVC was to pull preprocessed data files to local and remote machines for faster repository deployment. Throughout the project lifecycle, we used a couple of different version of our data files, consisting of different amounts of preprocessed data. The latest version contains the complete preprocessed WMD19 dataset for finetuning on the GPU assisted Cloud Compute environment.

### Question 11

> **Discuss your continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our continues integration into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We implemented a continuous integration setup using the Actions Function in Github. This included unit testing and linting, which was operable for multiple operating systems. Specifically, the tests were run on Python 3.12 and on Linux (Ubuntu), MacOS, and Windows. Unit testing was done using Pytest. For linting, Ruff was used, as mentioned before, which incorporated both Flake8 and iSort, adhering to Python standards PEP8 and handling import, sorting and replacing. In the future, MyPy could be added as a method to perform static type checking.

Here is a link to the Github actions:
https://github.com/MagnusBeng2/dtu_mlops_group_61/actions

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run an experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

For training the model, the chosen hyperparameters were by default taken from the Python script's Argparser. We found this to be easier than taking from a config file. Config files could be found under /src/models/config.

In the script, we set the default values in the argparsers, which could be overwritten when running the command: 'python ./src/models/train_model.py' --epochs 10 --lr 0.01'. This command line would configure the script to run with 10 epochs and a learning rate of 0.01 instead of the default values. The argsparser passed either the default or selected values to the wandb.init() function, whereafter the hyperparameters were loaded into the training script as follows:

config = wandb.config
    lr = config.lr
    epochs = config.epochs
    batch_size = config.batch_size
    seed = config.seed

Moreover, we utilized the Sweep functionality of WandB in sweep_model.py as to optimize hyperparameters, wherethrough the hyperparameter configuration was logged.

As for evaluate_model.py, it also utilized the argparsing functionality, taking the inputs 'seed', 'batch_size' and 'checkpoint_dir' for evaluation.

And for predict_model.py, it took an input and if the API was to be used, translating the input into a text in German.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:


In order to secure information and uphold reproducibility, both train_model.py and evaluate_model.py allowed for setting a seed before operation, ensuring same results could be obtained on two different devices and two different runs. Specifically, the two scripts would always take the default seed-value of 42, as is stated in the config. Additionally, we established docker images, which would ensure that models in the project could be identically run on all devices. And by logging the runs and their performances, we could inspect them side-by-side in WandB and confirm/disprove runs were in fact identical.


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

By logging the data in WandB, we were able to track the training loss for every run of train_model.py. Below is shown a training curve for a given run:

![Training loss](figures/train_loss.png)

Even though this specific run was run a small subset of the data, it is still clear that as the epochs progress, the training loss decreased.

In addition, we could also track the validation curve:

![Validation loss](figures/val_loss.png)

For the same run, it is clear that the model (in this run) exhibits some major overfitting, as the training and validation curve diverged from each other.

Utilizing logging, it would be possible for further studies to study and tinker with the parameters in the model, ultimately, achieving a satisfactory result.

Furthermore, our sweep_model.py allowed for conducting sweeps of the train_model.py, enabling a search for the best parameters.

In this simple sweep of the model and with 3 different learning rates, it was possible to deduct a learning rate that would give the lowest validation loss.

![Sweep hyperparameters](figures/hyperparams.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For most practical tests, containerized applications were not used. For deploying our model in the Cloud for inference, however, a docker was created with a minimalist setup including necessary packages for the predict_model.py script. The dockerfile used for this setup is called docker.dockerfile in our repository. To build and run the dockerfile locally the following commands can be used:

docker build . -f docker.dockerfile -t inference:latest
docker run -p 8080:8080 inference:latest


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When we executed the code locally for testing, we typically used the --debug_mode as a command option, meaning the datasize would be shrunk to 10% of the original size. When the code would produce errors in the terminal and not run, we would debug either using Copilot or ChatGPT to obtain fixes and recommendations. Furthermore, we would typically also insert simple print statements into the cpde, e.g., to check the size of tensors.

When we were able to run the code locally, first then could we begin to export the functionality to the GCP.

For profiling, we utilized the in-built tool from Pytorch Lightning to profile the training. However, we did not improve the coding based on the profiling, as we found it adequate in this project to just illustrate that profiling was possible. Nonetheless, the profiling did point us in directions on how to improve the code.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

The following GCS were used:

Buckets:
GCS Buckets were used to store the WMT19 dataset for training. The data inside the bucket was connected and version controlled by DVC.

Cloud Build:
The Image build by the dockerfile for inference using Cloud Run, was created using Cloud Build.

Cloud Build (Triggers):
A trigger was setup to automatically build a new image when changes to the repository were pushed. For minimizing usage in the future, the trigger should be modified to only update when the dockerfile is updated, not unrelated files.

Cloud Run:
For deploying inference for our model, Cloud Run is used as an easy, scalable solution. The Image created by Cloud Build in unison with FastAPI would be used to create a HTTP Endpoint.

Compute Engine:
The training was conducted in the Compute Engine using a Nvidia T5 GPU. The repository was cloned to the remote machine, with DVC providing the data set.


### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

During the project, we made extensive use of the Compute Engine for training and infering our model, yielding promising results for translation. The setup was simple: the remote machine was initialized with the | "pytorch-latest-cpu" image and the Nvidia T5 GPU. Our repository was transferred and kept up-to-date using Git. Training and inference was completed for parts of the data set. Two difficulties were experienced during use: (1) At multiple occasion, the SSH connection was lost. This would regularly happen at the 1h mark, however, changing session timeout settings did not help resolve this issue. This issue could be resolved using tmux in future session. (2) We did not implement a process for exporting trained weights and biases.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Our bucket consisted of the complete WMT19 data set, as well as 200,000 preprocessed samples.

![my_image](figures/ourbucket.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

The GCP Artifact Registry consisted of the inference-image that was automatically created using the docker.dockerfile connected to our repository. We were not able to access the Artifact Registry at this point because the billing account associated with the project ran out of credits. We tried associating two credit cards to a new billing account to access the Artefact Registry but both were declined.

![GCP Registry](figures/nope.png)


### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Build history](figures/build_history_cloud.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We were able to deploy our model locally using a FastAPI and Uvicorn framwork. Inference worked with a small delay compared to directly accessing the inference function. The framework for deploying the model in the cloud was to (1) define an inference model using an ONNX framwork, (2) creating an inference docker environment using the Artifact Registry, (3) use Google Cloud Run to create a scalable and efficient HTTP endpoint for serving requests.

We skipped the definition of our inference step as an ONNX model and went directly to deploying the model in the cloud. The docker was successfully created and run both locally and in the cloud, however, the credits ran out before we could debug the HTTP endpoint in Google Run.

Locally, the API can be called with the following CURL statement or in a browser.

curl -X GET "http://localhost:8080/translate/Hello%20world"

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring in this project. We would like to have implented cloud monitoring of our deployed model, allowing for measuring the model performance and thus, we could make sure it operated as expected over time. This tracking could include metrics like accuracy and precission, or f1 score to possibly detect model performance decrease. Moreover, because we worked with a translation model, and if it was to begin producing incorrect outputs more frequently, it could indicate data drifting which translating model are, in general, higly susceptible to.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

We used $50,03, the complete credits from one group member. The credits ran out on the last day. Of the $50, $49,30	were used on the Compute Engine. The multiple failed attempts (due to disconnections) had a significant cost as time was spent reconnecting and redoing computations.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![Graphical reprsentation of architecture](figures/graphical_representation_of_architecture.png)
The starting point of the diagram is our local Pytorch application, which we wrapped in the **Pytorch Lightning** framework. This served as the inital steps of creating the mlops pipeline. We version-controled our project using **git** and **Github**. A new environment can be initialized using either **Conda** or **pip**. We opted to use `pipreqs` for finding the package requirements of our project, which made for seamless instantiation of the projects *requirements.txt*. We utilized `wandb` in conjunction with **pytorch lightning** for logging the 'experiments'/ training of our *NLP* model. For training configuration `wandb` performed satisfactory, hence `hydra` was omited from this project. These are the essential parts which are contained into a **docker** container. Locally the project follows the codestructure of **Cookiecutter**.

In order to utilize the **GPC** git and dvc both provides a link from the local machine. Git furthermore enabled **Github actions** for testing the code before uploading to a remote storage. Using a **trigger** connected to the github repository we created **docker images** in **docker containers** in the cloud.

When training a dataset stored in a **GCP bucket** was utilized. Information sharing and version control of the dataset was handled by utilizing **dvc**. We interfaced with our application through **Cloud Run** by using the **Fast API** framework. Finally, we didn't utilize monitoring as we had plenty of work on our hands, trying to interface with and getting our model to run on cloud.

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The project lifecycle was marked by many small mistakes (and learnings). Most of the group members had little experience in GitHub. This resulted in, e.g., diverting version branches, which needed to be solved, which would be solved by using merge and git rebase. As the project scope grew, compliance with good practice project structure was neglected. The project in its current state had many features, but had little in common with the initial cookiecutter template. However, features were working and this problem could be resolved somewhat fast.

Initially, a lot of time was used on finding and understanding the nature of the data and how it was going to be processed. A significant amount of time was spent getting the different Python scripts inside of src/models running and making it fit to the processed data. As more features were introduced, such as logging and Wandb, it highlighted the importance of utlizing the practice of version control.

Another challenge was definitely to deploy the model to the cloud. This included: (1) understanding the GCS environment and navigating the GUI, (2) understanding the functionality and interconnection between the different services, (3) creating a deployable repository that supports the Google Cloud Services and finally (4) successfully deploying the model, while adhering to continous integration best practices.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:
During our project, we used Google Sheets to keep track of the project. This ensured that everybody knew what each group member was working on. Initial tasks such as environment setup were done in cooperation of all group members.

Peter A. Gründer s214987:
- Cookiecutter compliance
- Docker files
- Cloud deployment
- Compute Engine training
- API operation
- Cloud inference
- Pytest
- DVC

Alex J. Hagedorn s215002:
- Establishing and writing workflows
- Pre-commits
- Continuous integration
- Ensuring workflows worked correctly in Github


Magnus Bengtsson s216169:
- GitHub initializing
- Making the scripts for downloading and preprocessing the data
- Getting the src/models to work
- WandB and logging initializing
- Profiling and Sweeps
- Code coverage and PEP8 compliance

Student s224190:
- Existing
