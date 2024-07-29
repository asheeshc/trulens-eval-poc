import os
os.environ["OPENAI_API_KEY"] = ""

uw_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""

wsu_info = """
Washington State University, commonly known as WSU, founded in 1890, is a public research university in Pullman, Washington.
With multiple campuses across the state, it is the state's second largest institution of higher education.
WSU is known for its programs in veterinary medicine, agriculture, engineering, architecture, and pharmacy.
"""

seattle_info = """
Seattle, a city on Puget Sound in the Pacific Northwest, is surrounded by water, mountains and evergreen forests, and contains thousands of acres of parkland.
It's home to a large tech industry, with Microsoft and Amazon headquartered in its metropolitan area.
The futuristic Space Needle, a legacy of the 1962 World's Fair, is its most iconic landmark.
"""

starbucks_info = """
Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington.
As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
"""

import os
import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'),model_name="text-embedding-ada-002")

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
#import chromadb.utils.embedding_functions as embedding_functions
#openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'),model_name="text-embedding-ada-002")
openai_ef = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'),model_name="text-embedding-ada-002")
chroma_client = chromadb.Client()
# vector_store = chroma_client.get_or_create_collection(name="Washington", embedding_function=embedding_function)
vector_store = chroma_client.get_or_create_collection(name="Washington", embedding_function=openai_ef)

vector_store.add("uw_info", documents=uw_info)
vector_store.add("wsu_info", documents=wsu_info)
vector_store.add("seattle_info", documents=seattle_info)
vector_store.add("starbucks_info", documents=starbucks_info)

#vector_store = chroma_client.get_or_create_collection(name="Washington", embedding_function=embedding_function)

from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
tru = Tru()
tru.reset_database()

from openai import OpenAI
oai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(
            query_texts=query,
            n_results=4
        )
        # Flatten the list of lists into a single list
        return [doc for sublist in results['documents'] for doc in sublist]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=
        [
            {"role": "user",
            "content": 
            f"We have provided context information below. \n"
            f"---------------------\n"
            f"{context_str}"
            f"\n---------------------\n"
            f"Given this information, please answer the question: {query}"
            }
        ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion

rag = RAG_from_scratch()

from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.openai import OpenAI

import numpy as np

provider = OpenAI(model_engine="gpt-4o")

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve.rets[:])
    .aggregate(np.mean) # choose a different aggregation method if you wish
)

# from trulens_eval import TruCustomApp
# tru_rag = TruCustomApp(rag,
#     app_id = 'RAG v1',
#     feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])

# with tru_rag as recording:
#     rag.query("When was the University of Washington founded?")

# tru.get_leaderboard()

# last_record = recording.records[-1]

# from trulens_eval.utils.display import get_feedback_result
# feedback_results = get_feedback_result(last_record, "Context Relevance")
# print(feedback_results)

# note: feedback function used for guardrail must only return a score, not also reasons
f_context_relevance_score = (
    Feedback(provider.context_relevance, name = "Context Relevance")
)


from trulens_eval.guardrails.base import context_filter

class filtered_RAG_from_scratch:
    @instrument
    @context_filter(f_context_relevance_score, 0.75, keyword_for_prompt="query")
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(
        query_texts=query,
        n_results=4
    )
        return [doc for sublist in results['documents'] for doc in sublist]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=
        [
            {"role": "user",
            "content": 
            f"We have provided context information below. \n"
            f"---------------------\n"
            f"{context_str}"
            f"\n---------------------\n"
            f"Given this information, please answer the question: {query}"
            }
        ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query=query)
        completion = self.generate_completion(query=query, context_str=context_str)
        print(completion)
        return completion

filtered_rag = filtered_RAG_from_scratch()

from trulens_eval import TruCustomApp
filtered_tru_rag = TruCustomApp(filtered_rag,
    app_id = 'RAG v2',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])

with filtered_tru_rag as recording:
    filtered_rag.query(query="when was the university of washington founded?")