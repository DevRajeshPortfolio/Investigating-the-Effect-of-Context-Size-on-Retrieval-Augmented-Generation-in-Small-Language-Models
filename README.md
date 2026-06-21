Hello ! This is the set - up that i used to get data, and experiment for my research paper investigating the effect of changing the context size on RAG in Small
Language modeels (2.8B parameter model was used - I used Phi-2 by Microsoft )

Here is the file structure : 

rag-small-context
|
-main.py 
-requirements.txt 
-results.csv
-results_no_rag.csv 
-test.py 
-test_chunking.py 
-test_model.py 
-test_retreival.py 
-testing_dependencies.py 
---- Data 
     \ 
      -load_data.py 
      -__innit__.py 

 ---- models 
     \ 
      -load_model.py

---- Prompts
     \ 
      -context_controller.py
      -templates.py 
---- Results
     \ 
      -phi2_rag_results.csv
---- Evaluation
     \ 
      -metrics.py
      -evaluate_results.py 
      -__innit__.py 
---- Experiments
     \ 
      -run_experiment.py 
      -__innit__.py 
---- Retrieval 
     \ 
      -bm25_retreiver.py
      chunking.py
      -__innit__.py 
