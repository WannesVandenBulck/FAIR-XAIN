# FAIR-XAIN
FAIR-XAIN framework for fairness in explainable AI narratives

$ Data is prepped for RF ML models 
$ 4 datasets: law school, student, german credit and saudi employment
$ need to include logistic regression model also
$ need to make sure enough adverse predicted instances exist to generate XAINs for and compute fairness in later stages
$ prompts can be made for all datasets for both SHAPstories and CFstories generation 

# Walkthrough code 

$ 1. prep_{dataset}.ipynb files exist to prepare the datasets fit for random forest models. Random forest models are trained and evaluated. Also contain contextual information and information on features used in the prompts. Everything is saved under {dataset}_dataset folders, that thus contain (a) cleaned training and test set (parquet file), (b) trained RF model (pickle file) and (c) dataset info. 

$ 2. predictions.py uses the random forest machine learning models to predict whether an instance from any dataset belongs to the adverse class or not. The adversely predicted instances are saves as {dataset}_adverse.csv in the {dataset}_dataset folder. 


$ 3. explanations.py generates SHAP values based on the TreeSHAP algorithm and counterfactuals based on the NICE algorithm for all adversely predicted instances to explain why they belong to the "bad" class. Saved as {dataset}_shap.csv and {dataset}_cf.csv in the {dataset}_dataset folder under data under datasets_prep.  

$ 4. prompt_{dataset}.py files generate the prompt that will be fed to the LLM to generate XAINs. The prompt can be generated with the aim of producing SHAPstories and CFstories. 

$ 5. view_prompt.py can be ran to inspect a prompt so necessary adaptations can be made. A prompt is saved as prompt_{dataset}_{index}_s.txt for SHAP and prompt_{dataset}_{index}_c.txt for CF in the main branch. Only use locally to inspect. 

$ 6. make_narratives.py is used to generate the XAINs. The prompt made in prompt_{dataset}.py is passed to an LLM via API call and the results contain (in addition to the narrative) information on dataset, instance, prompt type, provider, model and timestamp. 

$ 7. view_narratives.py is used just as view_prompt.py to locally and quickly view a narrative to gain understanding of its contents and adapt the generation process or prompting strategy accordingly. The output is saved as a .txt file in the main branch. 