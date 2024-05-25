# DSPY - Programming Foundation Model - Inbuild chain of thought prompting


import os
import dspy
from dspy.datasets import HotPotQA
import dspy.evaluate
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dotenv import load_dotenv
from dsp.utils import deduplicate

# Load the environment variables
load_dotenv()

# Configure and Load Data
model = dspy.OpenAI(model="gpt-3.5-turbo")
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=model, rm=colbertv2)
dataset = HotPotQA(train_seed=1, train_size=20, dev_size=50, test_size=0, eval_seed=2023)

train_data = [x.with_inputs('question') for x in dataset.train]
test_data = [x.with_inputs('question') for x in dataset.dev]

# print(len(train_data), len(test_data))
# print(f"Train data: {train_data[0]}")
# print(f"Test data: {test_data[0]}")
# # print(dataset.train[0])


testing_data = test_data[18]
# print("\n Testing Question & Answer \n")
# print("Question: ", testing_data['question'])
# print("Answer: ", testing_data['answer'])


# Basic ChatBot 
class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(description="Answer to the question less than 10 tokens")

print("\n ---Response from the basic chatbot--- \n")
generated_answer = dspy.Predict(BasicQA)
pred = generated_answer(question=testing_data['question'])
print("\n predicted Answer: ", pred.answer)
print("\n Ground Truth: ", testing_data['answer'])

# Chain of thought prompting
print("\n ---Generate Response using Chain of thought prompting--- \n")

generate_answer_cot = dspy.ChainOfThought(BasicQA)
pred = generate_answer_cot(question=testing_data['question'])
model.inspect_history(n=1)
print("\n Answers predicted: ", pred.answer)
print("\n Ground Truth: ", testing_data['answer'])

#------------------------------------------------------------

print(" ---RAG Model---")

#Build chatbot with chain of thought and context --> RAG

# 1.Signature module
class DspyRAG(dspy.Signature):
    """
    Description: This is a Signature class for generating Answer
    """
    context = dspy.InputField(description="It may contain the relevant Facts")
    question = dspy.InputField()
    answer = dspy.OutputField(description="Answer to the question less than 10 tokens")

# 2. Module Pipeline
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(DspyRAG)
    
    def forward(self, question):

        # Retrieve the context
        context = self.retrieve(question).passages

        # Generate the answer
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(context = context, answer=prediction.answer)


# # 3. Optimizer
def validate_contex_and_answer(testing_data, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(testing_data, pred)
    answer_PM = dspy.evaluate.answer_passage_match(testing_data, pred)

    return answer_EM and answer_PM

# Initialize the Optimizer
teleprompter = BootstrapFewShot(metric=validate_contex_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=train_data)


# 4. Execute the pipeline
pred = compiled_rag(testing_data.question)
print("\n predicted Answers using DSPY RAG: ", pred.answer)
print("\n Ground Truth: ", testing_data.answer)


# 5. Evaluating the Answers
print("\n ---Evaluating the Answers--- \n")

# Uncompiled Baleen RAG (No Optimizer)
class GenerateSearchQuery(dspy.Signature): 
    """Generate search queries from questions."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    search_query = dspy.OutputField()

# The main Purpose of using Baleen is to automatically modify the question(prompt) and divide it into chunks. For Example, if the question is 
# "What is the capital of India?", Baleen will automatically modify the question to "What is the capital of India?" and 
# "India is the capital of which country?" and divide the question into chunks.

class BaleenRAG_non_optimizer(dspy.Module):
    def __init__(self, passage_per_hop=3, max_hops=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passage_per_hop)
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.generate_answer = dspy.ChainOfThought(DspyRAG)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).search_query
            passage = self.retrieve(query).passages
            context = deduplicate(context + passage)
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    
uncompiled_baleen_rag = BaleenRAG_non_optimizer()
pred = uncompiled_baleen_rag(testing_data.question)

print("\n Question: ", testing_data.question)
print("\n predicted Answers using Baleen RAG: ", pred.answer)
print("\n Ground Truth: ", testing_data.answer)

# TODO : Compiled Baleen RAG with Optimizer.
