import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma  # Updated from langchain_community.vectorstores
from langchain_huggingface import HuggingFaceEmbeddings  # Updated from langchain_community.embeddings
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Use the following context to answer the question concisely and accurately:

{context}

Question: {question}
Answer:
"""
def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    try:
        # Prepare for the DB
        # embedding_function = OpenAIEmbeddings()
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        try:
            from langchain_community.embeddings import FakeEmbeddings
            print("Falling back to FakeEmbeddings for testing purposes")
            embedding_function = FakeEmbeddings(size=384)  # Match the embedding size of the model

        except Exception:
            print("Could not load any embedding function. Exiting.")
            return


    # Search the DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 or results[0][1]< 0.4:
        print(f"Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(f"##### context_text:{context_text}")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(f"###### prompt:{prompt}")
    
    #############################
    # Using DeepseekR1 model
    # https://github.com/marketplace/models/azureml-deepseek/DeepSeek-R1/playground/code
    #############################
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage
    from azure.ai.inference.models import UserMessage
    from azure.core.credentials import AzureKeyCredential

    # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings. 
    # Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    client = ChatCompletionsClient(
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.environ["DEEPSEEKR1_GITHUB_TOKEN"]),
    )

    try:
        response_text = client.complete(
            messages=[
                UserMessage(prompt),
            ],
            model="DeepSeek-R1",
            max_tokens=1000,
        )
        print(f"###### response_text:{response_text.choices[0].message.content}")
    except Exception as e:
        print(f"Error during API call: {e}")
        return
    
    # model = ChatOpenAI()
    # response_text = model.predict(prompt)

    #############################
    # Using the local model
    #############################
    # # Load a pre-trained model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace "gpt2" with a larger model if needed
    # tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token
    # model = AutoModelForCausalLM.from_pretrained("gpt2")

    # # Generate a response using the local model
    # # inputs = tokenizer(prompt, return_tensors="pt")
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # # outputs = model.generate(inputs["input_ids"], max_length=500, num_return_sequences=1)
    # outputs = model.generate(
    # inputs["input_ids"],
    # attention_mask=inputs["attention_mask"],  # Explicitly pass the attention mask
    # max_new_tokens=200,  # Limit the number of tokens to generate
    # num_return_sequences=1,
    # pad_token_id=tokenizer.eos_token_id  # Set the pad token to eos_token_id to avoid warnings
    # )
    # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response_text = "\n".join(set(response_text.split("\n")))  # Remove duplicate lines
    #####################
    # End of local model
    #####################

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.choices[0].message.content}\n Sources: {sources}"
    print(formatted_response)



if __name__ == "__main__":
    main()