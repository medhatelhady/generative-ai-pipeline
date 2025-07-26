from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os

os.environ['OPENAI_API_KEY'] = "voc-1422108720153663203699167c098bcbd5853.29117029"
os.environ['OPENAI_BASE_URL'] = "https://openai.vocareum.com/v1"


with open("search_result.txt", 'r') as f:
    results_text = f.read()

# Node 3: Summarize search results in patient-friendly language
def summarize_health_info(results_text):
    

    # === 2. Split into chunks ===
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100
    )
    chunks = splitter.split_text(results_text)

    # === 3. Setup LangChain LLM Chain ===
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")  # or use "gpt-3.5-turbo" etc.

    prompt_template = PromptTemplate.from_template("""
    You are a dataset generator for training a language model.
    Given the following content, extract all relevant and possible question-answer pairs that could help teach this content. 
    Focus on diversity, clarity, and full coverage. THE ANSWER MUST BE SPAN OF TEXT OF THE CONTEXT WITHOUT EDITING.

    Content:
    {chunk}

    Return the output in the following JSON format:
    [
    {{ "question": "...", "answer": "..." }},
    ...
    ]
    """)

    chain = prompt_template | llm

    # === 4. Run on All Chunks ===
    qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        response = chain.invoke({"chunk": chunk})
        #print(response)
        try:
            chunk_qas = json.loads(response.content)
            for i in range(len(chunk_qas)):
                chunk_qas[i]['context'] = chunk
            qa_pairs.extend(chunk_qas)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON in chunk {i+1}")
            continue


    return qa_pairs


qa_pairs = summarize_health_info(results_text)

for i in range(len(qa_pairs)):
    if qa_pairs[i]['answer'].lower().strip('.') in qa_pairs[i]['context'].lower():
        answer = qa_pairs[i]['answer']
        answer_start = qa_pairs[i]['context'].lower().index(qa_pairs[i]['answer'].lower().strip('.'))
        qa_pairs[i].pop('answer')
        qa_pairs[i]['answers'] = {"text": [answer], 'answer_start': [answer_start]}

# === 5. Save Output ===
with open("qac_pairs.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)