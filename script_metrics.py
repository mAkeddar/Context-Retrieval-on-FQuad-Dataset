import functions as f

def main(): 
    contexts,questions,answers = f.read_from_terminal()

    joined_frames = f.join_dataframes(contexts,questions)

    contexts = f.processC(contexts)

    questions = f.processQ(questions)
    nb_questions = len(questions)
    print(nb_questions)
    V,Tf = f.tfifd_components(contexts['Contexts'])
    
    top_1_accuracy, mean_retrieval_time = f.top_1_accuracy(len(questions.index),joined_frames,Tf,V,contexts)
    top_4_accuracy = f.top_4_accuracy(len(questions.index),joined_frames,Tf,V,contexts)
    mean_rank = f.mean_rank(nb_questions,joined_frames,Tf,V,contexts)
    
    print("\nTop_1 accuracy = "+ str(top_1_accuracy) + " Mean retrieval time = "+ str(mean_retrieval_time))
    print("\nTop_4 accuracy = "+ str(top_4_accuracy))
    print("\nMean rank for correct answer = "+ str(mean_rank)) 


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram exited by user !")