from transformers import pipeline


question = "What type of vehicles are the Chrysler TEVan, Ford Ranger EV pickup truck, GM EV1 and S10?"
context = "goal being a move to zero-emissions vehicles such as electric vehicles.( In response, automakers developed electric models, including the Chrysler TEVan, Ford Ranger EV pickup truck, GM EV1 and S10"

model_path = "my_awesome_peft_qa_model"


question_answerer = pipeline("question-answering", model=model_path)
question_answerer(question=question, context=context)

print(answer['answer'])
