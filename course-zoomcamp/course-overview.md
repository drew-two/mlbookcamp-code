# Machine Learning Zoomcamp
## September 5th, 2022

- Example slide - linear regression
	- This is the extent of the math.
	- We will do matrix multiplication

## Book
Based on book Alexey wrote - Machine Learning Bookcamp w/ O'Reilly
	- Had to rework some things to make up-to-date

## Course Team
- Alexey Grigorev - Lead Data Scientist at OLX Group, Germany
- Timur Kameliev (TA) - Took all the courses prior
- Thinam Tamang (TA) - Lots of "Learning in public"

## Is it for me?
- Pre-requisites
	- Experience with programming
	- You know Python or can pick it up quickly
	- Being comfortable with command line (Git, etc)

- Not required
	- Previous experience with ML

## Syllabus 
- Github repo: https://github.com/alexeygrigorev/mlbookcamp-code
	- Asks that you star the repo

- Full table of contents in Readme
	- Links to each individual sub chapter

## Table of Contents
1. Introduction
	- Formal definition of ML: g(X) ~ y
		- g is the model, X is the features, y is the target
	- Process - where Data Science and Machine Learning fit into the SWE process
	- Math - linear algebra from a developer's point of view
		- Will implement matrix multiplication

2. Linear Regression
	- Problem: Car price prediction
		- Course work will be project-based
		- Will gradually work through project, explaining everything needed along the way
	- Will implement things from scratch ourselves
		- Shows us what libraries are doing under the hood
		- Only time when we will do this
3. Logistic Regression
	- Problem: Telecom company churn prediction
		- May examine user data to touch base with users before they leave
		- Build model to predict customers that might leave and send promotional email
	- Will not be implementing these ourselves - using scikit-learn
4. Model Evaluation
	- Abstract, may need to rewatch this a couple of times
	- Problem: Evaluating churn prediction model
	- Will talk about metrics like accuracy, precision, recall, AUC
5. Model Deployment
	- Need to learn how to deploy model; model is ultimately useless if you do not deploy it
	- Put it early, right after the basics on purpose
	- Talk about containerization and flask
	- Since this class is more about MLE than DS, we focus on deployment
	- Will use some tools specifically for this, called BentoML
6. Tree-based Models
	- Problem: Credit risk analysis
		- Banks want to know if someone is likely to pay back a loan
		- Predicting if someone might default on a loan
	- More complex models, so we do this after deployment, but also **more powerful**
7. Midterm Project
	- Need to put everything covered so far into practice
		- Finding a problem and an applicable dataset
		- Describing problem and explaining how a model could be used
		- Preparing data, EDA, finding and engineering features
		- Training multiple models, tuning their performance and selecting the best model
		- Exporting the notebook into a script
		- Putting model into a web service and deploying locally with docker
		- BONUS: deploying to cloud
8. Deep Learning
	- Project: Neural networks for image classifcation w/ Keras
		- Student PyTorch code available from class last year
	- Will be building this model

9. Serverless
	- Project: Deploy chapter 8 model with AWS Lambda

10. Kubernetes
	- Project: Deploy chapter 8 model with Kubernetes
	- Very different approach from Lambda
		- Kubernetes is much more heavyweight than Lambda
	- Very common model in industry

11. (Optional) KServer
	- System on top of Kubernetes that simplifies things
	- Optional as it is very complicated to install

After that:
	- Capstone 1
	- Capstone 2
	- Article
Think of these of having two capstone projects and you decide which one to take. Can take both though.

Course will take 4-5 months, with course content taking about 4 months. Content concludes December, projects conclude in January.

## Course Logistics
- Lessons 
	- Pre-recorded
	- Posted on (Youtube Channel)[https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR]
- Office hours
	- Live on Mondays 17:00 CEST/11:00 EST
	- Answer questions
	- Homework solutions will be recorded separately
- Project
	- 2 weeks - working on the project
	- 3rd week - peer reviewing
		- Approach this not like a duty, but an opportunity to learn
	- Most topics will be 1 week, except deep learning, which is 2 weeks
		- Deep learning done in parallel with peer review

## FAQ
- Available in (Google Docs)[https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit#heading=h.98qq6wfuzeck]
	- Always check FAQ first before asking Slack
	- When you get the answer, put in FAQ

## Homework
- Submission done via Google Forms
	- Watch videos for the week then check cohorts folder for your year
	- **Homework deadline**: Monday after week at 23:00 CEST/17:00 EST

## Learning in Public
- Submission entry in every homework form
- **Strongly suggested** to share each weeks work on LinkedIn, Twitter or your blog
	- If you do one every day, you can submit 7 links for 7 max points
	- Separate by space or line
- Leaderboard for marks
	- Each homework assignment and project has a leaderboard tab, as well as an overall leaderboard
	- Can opt into Top 100 students after course has finished; displayed on Github and advertises student
- Learning in Public story - Michael's LinkedIn
	- Boss's thought that maybe he was going to leave company because of his learning in public
	- Ended up getting a raise
- Learning in public can be almost as hard as the homework, worth it to step outside of comfort zone
	- Can also lead to networking

## Certificate
Given on condition of passing 2 out of 3 projects (project 1, 2, or capstone)
- Homework not actually required
- Can still get certificate if you join late

## Timecodes
- Github issue for each of the videos automatically created
- Comment that you are taking it first
- Comment timecodes for each topic in videos so useful Youtube chapters can be made

## Slack
- Part of greater DataTalks.Club community
- Only talk about course in #ml-zoomcamp

## Sponsors
- BentoML - company behind specialized tool for serving ML models
- One thing they are looking for is how many people join the BentoML slack after course ends
	- Please join if you can - https://join.slack.bentoml.org/
- Can sponsor Alexey directly on GitHub

## Q & A
- Employability?
	- Only anecdotes. Interview on DataTalks.Club
- Why is it called Zoomcamp?
	- Original book was called *Machine Learning Bookcamp*
	- When creating course on the book, *Zoomcamp* seemed like a fun play on words
	- Zoom actually cannot accomdate more than 300 watchers cheaply
		- So it was switched to Youtube, but the name stuck
	- Now *Zoomcamp* is kind of a brand
- Can I complete it early in 1-2 weeks?
	- Maybe, but homework will not all be released in time.
- Can we add this to our resume?
	- By all means
- Is the textbook a recommended complementary reading?
	- Yes, but not required. You can buy it if you want to support Alexey
	- **KServe** information however is outdated
- Difficulty?
	- From the last cohort:
		- ~5,500 joined
		- ~400 finished the first homework
		- ~100 got the certificate
	- Around 1/4th of the people who did the homework actually finished
	- Pretty demanding in terms of time.
		- Can do course at your own pace, and catch up and do project at the end.
- Is there a course on DataTalks.Club on NLP?
	- Not yet
- Any office hours besides Monday?
	- No, but can add questions in advance
- Any plans for a deep learning zoomcamp in the future?
	- No, not Alexey's domain
	- Maybe, if there is a lot of domain and the right instructors
- Which book would you suggest for theoretical ML to follow up with this course and in general?
	- Don't need any theoretical book for this course
	- One possibility is **Elements of Statistical Learning** - math-heavy!
- Is the ML Engineering Roadmap ready?
	- The course is the ML Engineering Roadmap
- Cloud credits given for the course?
	- AWS recently gave Alexey some, but he will make them hackathon or competition prizes
- Can we use Colab?
	- Yes, but AWS and deployments cannot be done in Colab
- Can I learn Github on the go?
	- Yes
- Any video for setting the environment from the course?
	- See MLOps-zoomcamp module 1, video 6. Setup is very similar
- Is Docker necessary?
	- Yes. If not possible, use it in the Cloud
- What Python knowledge is necessary?
	- See module 1
- Passing score?
	- Don't worry, if you put effort you will pass
- Should we know MLOps tools such as Kubernetes, Docker, CI/CD
	- Docker not an MLOps tool but very necessary
- Can we make study groups?
	- Please do
- Minimum effort per week needed?
	- 10 hours
- Slack vs Telegram?
	- Slack is for the community, Telegram is for announcements
	- Telegram is optional, each post automatically also posts to Slack
- Minimum system requirements?
	- Don't need any particularly powerful computer, just need internet
	- Can use GCP or AWS. GCP gives $300 of free credits at account creation, AWS has free tier
- Is it necessary to go into the math of ML?
	- Not for the course
	- You WILL if you plan to be a data scientist
	- As an ML Engineer, unlikely, but maybe depending on the company
- Will there be videos on how to be more profitable, or ML interview preparation?
	- See https://DataTalks.Club, see Podcast and career topics posts.
	- Lots of new content all the time
