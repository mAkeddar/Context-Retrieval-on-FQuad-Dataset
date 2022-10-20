import functions as f
import time

def main(): 

    contexts,questions,answers = f.read_from_terminal()

    contexts = f.processC(contexts)

    while(True):
        question = f.read_question()

        question = [question.replace('?','')]

        V,Tf = f.tfifd_components(contexts['Contexts'])

        nb_contexts = int(f.read_nb_contexts())

        print(question[0]+ "?")

        most_similar_context_indices, cosine_similarities = f.calculate_similarity(Tf,V,question,nb_contexts)

        start_time = time.time()
        f.show_retrieve_context(most_similar_context_indices,contexts['Contexts'],1)
        end_time = time.time()-start_time
        print("Retrieval time = "+ str(end_time)+"s" )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram exited by user !")