This README contains informations to launch the scripts for context retrieval on FQuad Dataset : 

- Requirements : 
	Please install pandas and sklearn in your python environment.
	Download the train and validation dataset from : https://fquad.illuin.tech/
	
- script_test.py :

	- This script retrieves the k top contexts and prints them on the terminal. To run it, please go to your terminal and write : python3 script_test.py path_to_your_dataset
	- Then you'll be asked to enter a query and the number of contexts needed.
	- Then the script prints the retrieved contexts in this format : rank x context :"..."
								         rank y context :"..."
									 Retrieval_time : XXXs

- script_metrics.py : 
	
	- This script computes the 3 metrics defined for this project : top_1 accuracy, top _4 accuracy and mean rank. 
	- To launch the script : python3 script_metrics.py path_to_your_dataset
	- The script prints the metrics on the terminal (It takes some times)

- functions.py : 
	- Python file containing the necessary function for this project
 


- Metrics : 				train				validation

	Top_1 accuracy : 		48%				59%
	Top_4 accuracy : 		70%				80%
	Mean_retrieval time : 		8ms				2ms
	Mean rank 	: 		68				10
