from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.7, top_p=0.9
)
llm = HuggingFacePipeline(pipeline=pipe)



# Use a compatible embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the FAISS vector store
vector_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)


#Create a retriever
retriever: BaseRetriever = vector_store.as_retriever(search_kwargs={"k": 5})



print('creating template')

prompt_template= """
You are an AI assistant. Answer the question based on the provided context if the question and context are related. Else generate the response solely based on the query. Do not give the prompt or context in your response instead just give the answer

{context}
Question : {question}

Answer 

"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Combines all retrieved documents into a single input for the LLM
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)

query = 'what is Data science'

def generate_response(query):
    text = qa_chain.invoke(query.strip())
    response = '\n'.join(text['result'].split('Answer')[-1].split('\n'))
    return response


print(generate_response(query))